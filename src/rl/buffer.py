"""Experience buffer for GRPO on-policy training.

Adapted from tiny-grpo (open-thought/tiny-grpo) — simplified for our
character-level arithmetic model with no HuggingFace dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Experience:
    """One rolled-out group of G completions for a single question.

    Tensor shapes  (G = group_size, T = max seq len, K = max answer len):
        sequences      [G, T]   full token IDs (prompt + answer, right-padded)
        old_log_probs  [G, K]   per-token log π_θ_old at rollout time
        ref_log_probs  [G, K]   per-token log π_ref  (frozen reference model)
        advantages     [G]      group-normalised advantage per rollout
        action_mask    [G, K]   True for real answer tokens (False = padding)
        returns        [G]      raw scalar rewards
        prompt_len     int      number of prompt tokens (same across the group)
    """
    sequences:     torch.Tensor  # [G, T]
    old_log_probs: torch.Tensor  # [G, K]
    ref_log_probs: torch.Tensor  # [G, K]
    advantages:    torch.Tensor  # [G]
    action_mask:   torch.Tensor  # [G, K]  bool
    returns:       torch.Tensor  # [G]
    prompt_len:    int

    def to(self, device: torch.device) -> Experience:
        return Experience(
            sequences=self.sequences.to(device),
            old_log_probs=self.old_log_probs.to(device),
            ref_log_probs=self.ref_log_probs.to(device),
            advantages=self.advantages.to(device),
            action_mask=self.action_mask.to(device),
            returns=self.returns.to(device),
            prompt_len=self.prompt_len,
        )


def pad_to_same_length(
    tensors: list[torch.Tensor],
    pad_value: float = 0.0,
    side: str = "right",
) -> torch.Tensor:
    """Pad a list of 1-D tensors to the same length and stack into a 2-D tensor."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            padding = (0, pad_len) if side == "right" else (pad_len, 0)
            t = F.pad(t, padding, value=pad_value)
        padded.append(t)
    return torch.stack(padded, dim=0)


class ReplayBuffer:
    """Simple in-memory buffer — cleared after every GRPO step (on-policy)."""

    def __init__(self) -> None:
        self._items: list[Experience] = []

    def append(self, exp: Experience) -> None:
        self._items.append(exp)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
