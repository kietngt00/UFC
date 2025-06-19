from torch import nn
from einops import rearrange
import torch
import math
import torch.nn.functional as F
# from diffusers.models.normalization import AdaLayerNormZero
from typing import Dict, Optional, Tuple

from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from einops import repeat


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, input_dim:int ,embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(input_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa
        

class CrossAttention1(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, head=8, hidden_dim=768, time_dim=1280):
        super().__init__()

        self.head = head
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // head

        self.to_q = nn.Linear(dim_q, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim_k, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim_v, hidden_dim, bias=False)
        self.to_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.pre_ln_q = nn.LayerNorm(dim_q)
        self.pre_ln_k = nn.LayerNorm(dim_k)
        self.norm1 = AdaLayerNormZero(time_dim, dim_v)

        self.ln = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()    

        self.zero_mlp = nn.Linear(dim_v, dim_v, bias=False)
        with torch.no_grad():
            self.zero_mlp.weight.fill_(0)

    def get_attn_map(self, q, k):
        """
        Args:
            q: [(B T), (S1 n), d]
            k: [(B T), (S2 n), d]
            v: [(B T), (S2 n), d]
        """
        q = self.pre_ln_q(q)
        k = self.pre_ln_k(k)

        Q = self.to_q(q)
        K = self.to_k(k)

        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.head)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.head)

        A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B_q, h, N, B_sp * N]
        A = A.softmax(dim=-1) 

        return A
    

    def forward(self, q, k, v, temb,):
        """
        Args:
            q: [(B T), (S1 n), d]
            k: [(B T), (S2 n), d]
            v: [(B T), (S2 n), d]
        """
        q = self.pre_ln_q(q)
        k = self.pre_ln_k(k)
        v, gate_msa = self.norm1(v, emb=temb)

        Q = self.to_q(q)
        K = self.to_k(k)
        V = self.to_v(v)

        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.head)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.head)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.head)

        O = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        O = rearrange(O, "b h n d -> b n (h d)")

        # Attn Residual connection: V & O shape miss match in cross attn --> Residual using O
        hidden_states = O + gate_msa.unsqueeze(1) * self.activation(self.to_o(O))

        return self.zero_mlp(hidden_states)