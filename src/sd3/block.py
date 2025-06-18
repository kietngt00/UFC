import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from diffusers.models.attention import JointTransformerBlock, FeedForward
from torch import nn

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int, task_indices: torch.Tensor = None):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if task_indices is not None:
        ff_output = torch.cat(
            [ff_forward(ff, hid_slice, task_indices) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )
    else:
        ff_output = torch.cat(
            [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
            dim=chunk_dim,
        )
    return ff_output


def ff_forward(
    self: FeedForward,
    hidden_states: torch.FloatTensor,
    task_indices: torch.FloatTensor,
):  
    hidden_states = self.net[0].proj(hidden_states, task_indices)
    hidden_states = self.net[0].gelu(hidden_states)
    hidden_states = self.net[1](hidden_states)
    hidden_states = self.net[2](hidden_states, task_indices)
    return hidden_states


def attn_forward( # JointAttnProcessor2_0
    attn: Attention,
    hidden_states: torch.FloatTensor,
    task_indices: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.FloatTensor:

    batch_size = hidden_states.shape[0]

    # `sample` projections.
    query = attn.to_q(hidden_states, task_indices)
    key = attn.to_k(hidden_states, task_indices)
    value = attn.to_v(hidden_states, task_indices)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, task_indices)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def block_forward(
    self: JointTransformerBlock,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    task_indices: torch.FloatTensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
):  

    joint_attention_kwargs = joint_attention_kwargs or {}
    if self.use_dual_attention:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=temb
        )
    else:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    # Attention.
    if task_indices is None:
        attn_output = self.attn(hidden_states=norm_hidden_states)
    else:
        attn_output = attn_forward(
            self.attn,
            hidden_states=norm_hidden_states,
            task_indices=task_indices,
            **joint_attention_kwargs,
        )

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    if self.use_dual_attention:
        if task_indices is None:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
        else:
            attn_output2 = attn_forward(self.attn2, hidden_states=norm_hidden_states2, task_indices=task_indices, **joint_attention_kwargs)
        attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
        hidden_states = hidden_states + attn_output2

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, task_indices=task_indices)
    else:
        if task_indices is None:
            ff_output = self.ff(norm_hidden_states)
        else:
            ff_output = ff_forward(self.ff, norm_hidden_states, task_indices=task_indices)

    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output

    return hidden_states
