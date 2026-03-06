"""
GRPO Loss — Group Relative Policy Optimization.

Faithful implementation of DeepSeekMath (Shao et al., 2024) Equation 3:

  J_GRPO(θ) = E_{q~p(Q), {o_i}~π_θ_old(·|q)} [
      1/G  Σ_i   1/|o_i|  Σ_t
        min( r_{i,t} · Â_i,  clip(r_{i,t}, 1-ε, 1+ε) · Â_i )
        − β · KL( π_θ || π_ref )_{i,t}
  ]

  where  r_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θ_old(o_{i,t} | q, o_{i,<t})
         Â_i     = (reward_i − mean(rewards)) / (std(rewards) + ε)

KL estimator: unbiased k3 approximation (Schulman, 2020):

  KL(π_θ || π_ref) ≈ r − log r − 1   where r = π_ref / π_θ

  Properties: always ≥ 0, equals 0 iff π_θ = π_ref.
  Cheaper than exact KL — no extra forward pass needed.

References:
  DeepSeekMath:     https://arxiv.org/abs/2402.03300
  KL approximation: http://joschu.net/blog/kl-approx.html
  tiny-grpo:        https://github.com/open-thought/tiny-grpo
"""

from typing import Optional

import torch
import torch.nn as nn


# ── Utility ───────────────────────────────────────────────────────────────────

def masked_mean(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = -1,
) -> torch.Tensor:
    """Mean over `dim`, counting only positions where mask=True."""
    if mask is None:
        return x.mean(dim=dim)
    m = mask.float()
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp(min=1e-8)


def approx_kl_divergence(
    log_probs:     torch.Tensor,           # log π_θ      [G, K]
    ref_log_probs: torch.Tensor,           # log π_ref    [G, K]
    action_mask:   Optional[torch.Tensor],  # bool         [G, K]
) -> torch.Tensor:
    """k3 unbiased KL estimate:  KL ≈ r − log r − 1  where r = π_ref / π_θ."""
    log_r = ref_log_probs.float() - log_probs.float()   # log(π_ref / π_θ)
    kl = log_r.exp() - log_r - 1.0                   # always ≥ 0
    if action_mask is not None:
        kl = kl * action_mask.float()
    return kl  # [G, K]


# ── Loss module ───────────────────────────────────────────────────────────────

class GRPOLoss(nn.Module):
    """GRPO actor loss (DeepSeekMath Eq. 3).

    Args:
        clip_eps  : ε — PPO clipping range  (default 0.2)
        kl_weight : β — KL penalty coefficient  (default 0.01)
    """

    def __init__(self, clip_eps: float = 0.2, kl_weight: float = 0.01, entropy_weight: float = 0.01) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight

    def forward(
        self,
        log_probs:     torch.Tensor,   # [G, K]  current policy
        old_log_probs: torch.Tensor,   # [G, K]  policy at rollout time
        ref_log_probs: torch.Tensor,   # [G, K]  frozen reference policy
        advantages:    torch.Tensor,   # [G]     group-normalised advantages
        action_mask:   torch.Tensor,   # [G, K]  bool
        # [G, K, vocab_size]  for entropy bonus
        full_log_probs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss : scalar — minimise this
            kl   : scalar — for logging only
        """
        # Broadcast advantages [G] → [G, K]
        adv = advantages.unsqueeze(-1).expand_as(log_probs)

        # Policy ratio  r = π_θ / π_θ_old
        # [G, K]
        ratio = (log_probs - old_log_probs).exp()

        # Clipped surrogate  (PPO-style)
        surr_unclipped = ratio * adv
        surr_clipped = ratio.clamp(
            1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr_unclipped,
                                 surr_clipped)          # [G, K]

        # KL penalty
        kl = approx_kl_divergence(
            log_probs, ref_log_probs, action_mask)   # [G, K]

        # Entropy bonus: H(π) = -Σ p log p  (maximize → negate untuk loss)
        entropy_loss = 0.0

        if full_log_probs is not None and self.entropy_weight > 0:
            entropy = -(full_log_probs.exp() *
                        full_log_probs).sum(dim=-1)  # [G, K]
            entropy_loss = -self.entropy_weight * \
                masked_mean(entropy, action_mask, dim=-1).mean()

        # Combined objective
        total = policy_loss + self.kl_weight * \
            kl                          # [G, K]

        # Mean over real answer tokens, then mean over rollouts
        loss = masked_mean(total, action_mask, dim=-
                           1).mean() + entropy_loss           # scalar
        # scalar (for logging)
        kl_mean = masked_mean(kl,    action_mask, dim=-1).mean()

        return loss, kl_mean
