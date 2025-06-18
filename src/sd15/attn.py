import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union, Optional, Dict, Any, Callable, Tuple
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int, task_indices):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff_forward(ff, hid_slice, task_indices) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def ff_forward(self, norm_hidden_states, task_indices):
    hidden_states = self.net[0].proj(norm_hidden_states, task_indices)
    hidden_states, gate = hidden_states.chunk(2, dim=-1)
    hidden_states = hidden_states * self.net[0].gelu(gate)
    hidden_states = self.net[1](hidden_states)
    hidden_states = self.net[2](hidden_states, task_indices)
    return hidden_states


def attn_forward(
        attn,
        hidden_states: torch.Tensor,
        task_indices: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states, task_indices)

    is_cross_attn = True 
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
        is_cross_attn = False
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        is_cross_attn = True

    key = attn.to_k(encoder_hidden_states, None if is_cross_attn else task_indices)
    value = attn.to_v(encoder_hidden_states, None if is_cross_attn else task_indices)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states, task_indices)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def block_forward(
    self,
    hidden_states: torch.Tensor,
    task_indices: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = attn_forward(
        self.attn1,
        norm_hidden_states,
        task_indices,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 1.2 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = attn_forward(
            self.attn2,
            norm_hidden_states,
            task_indices,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, task_indices)
    else:
        ff_output = ff_forward(self.ff, norm_hidden_states, task_indices)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states


def transformer_forward(
    self,
    hidden_states: torch.Tensor,
    task_indices: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Dict[str, torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
):
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
    #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
    #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None and attention_mask.ndim == 2:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    if self.is_input_continuous:
        batch_size, _, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states, inner_dim = _operate_on_continuous_inputs(self, hidden_states, task_indices)
    elif self.is_input_vectorized:
        hidden_states = self.latent_image_embedding(hidden_states)
    elif self.is_input_patches:
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
        )

    # 2. Blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states = torch.utils.checkpoint.checkpoint(
                block_forward(
                    block,
                    hidden_states,
                    task_indices,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            )
        else:
            hidden_states = block_forward(
                block,
                hidden_states,
                task_indices,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

    # 3. Output
    if self.is_input_continuous:
        output = _get_output_for_continuous_inputs(
            self,
            hidden_states=hidden_states,
            task_indices= task_indices,
            residual=residual,
            batch_size=batch_size,
            height=height,
            width=width,
            inner_dim=inner_dim,
        )
    elif self.is_input_vectorized:
        output = self._get_output_for_vectorized_inputs(hidden_states)
    elif self.is_input_patches:
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            embedded_timestep=embedded_timestep,
            height=height,
            width=width,
        )

    if not return_dict:
        return (output,)


def _operate_on_continuous_inputs(self, hidden_states, task_indices=None):
    batch, _, height, width = hidden_states.shape
    hidden_states = self.norm(hidden_states)

    if not self.use_linear_projection:
        hidden_states = self.proj_in(hidden_states, task_indices)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
    else:
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states, task_indices)

    return hidden_states, inner_dim


def _get_output_for_continuous_inputs(self, hidden_states, task_indices, residual, batch_size, height, width, inner_dim):
    if not self.use_linear_projection:
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        )
        hidden_states = self.proj_out(hidden_states, task_indices)
    else:
        hidden_states = self.proj_out(hidden_states, task_indices)
        hidden_states = (
            hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        )

    output = hidden_states + residual
    return output