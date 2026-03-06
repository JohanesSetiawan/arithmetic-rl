import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import apply_rope


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.

    Uses a single fused QKV projection (one matmul instead of three).
    Delegates to F.scaled_dot_product_attention, which automatically uses
    FlashAttention on CUDA or an optimised CPU kernel otherwise.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads  = config.n_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj  = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout   = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Fused QKV → split
        q, k, v = self.qkv_proj(x).split(C, dim=-1)

        # Reshape to [B, H, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # Merge heads → [B, T, C]
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))
