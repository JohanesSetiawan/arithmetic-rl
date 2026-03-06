import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .ffn import SwiGLUFFN


class DecoderBlock(nn.Module):
    """Single decoder-only transformer block using pre-norm architecture.

    Pre-norm (normalize *before* each sublayer) is more stable than post-norm
    when training from random initialization.

    Layout:  x → Norm → Attention → +x → Norm → FFN → +x
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = nn.RMSNorm(config.d_model)
        self.attn      = CausalSelfAttention(config)
        self.norm_ffn  = nn.RMSNorm(config.d_model)
        self.ffn       = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn(self.norm_ffn(x))
        return x
