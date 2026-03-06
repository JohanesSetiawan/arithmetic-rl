import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Positional Embedding (RoFormer, Su et al. 2021).

    Encodes position by rotating Q and K vectors in 2-D sub-spaces.
    Requires no fixed max sequence length — computed dynamically per forward pass.

    Inverse frequencies:  θ_i = base^(−2i / head_dim)
    """

    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        """Returns (cos, sin) rotation tables of shape [seq_len, head_dim]."""
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)          # [T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                # [T, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by 90°: [x1, x2] → [−x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,   # [B, H, T, head_dim]
    k: torch.Tensor,
    cos: torch.Tensor, # [T, head_dim]
    sin: torch.Tensor,
) -> tuple:
    """Apply RoPE rotation to queries and keys."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]  broadcast
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
