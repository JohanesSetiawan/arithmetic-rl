import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    SwiGLU(x) = SiLU(gate(x)) ⊙ up(x),  then projected down.
    Empirically outperforms ReLU/GELU FFN at equal parameter budget.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up   = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
