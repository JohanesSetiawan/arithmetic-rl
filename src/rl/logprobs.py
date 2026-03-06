"""Gradient-enabled log-probability computation for the GRPO update step.

Distinct from environment.py which computes log probs at rollout time (no_grad).
This module is called *during* the update, so gradients must flow through here.
"""

import torch
import torch.nn.functional as F

from ..model.transformer import ArithmeticTransformer


def compute_answer_log_probs_batch(
    model:       ArithmeticTransformer,
    sequences:   torch.Tensor,   # [G, T]  prompt + answer, right-padded with 0
    # number of prompt tokens (same for all in batch)
    prompt_len:  int,
    # [G, K]  bool — which answer positions are real
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through π_θ; return log probs of answer tokens.

    Token indexing:
        logits[:, i, :]  predicts token at position  i + 1
        Answer tokens start at position  prompt_len
        → predicted by logits at positions  prompt_len-1 … T-2

    Returns:
        log_probs : [G, K]   (K = T − prompt_len, including padding positions)
    """
    logits = model(sequences).float(
    )                           # [G, T, vocab_size]

    answer_logits = logits[:, prompt_len -
                           1: -1, :]          # [G, K, vocab_size]
    answer_ids = sequences[:, prompt_len:]                   # [G, K]

    # [G, K, vocab_size]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    # [G, K]
    gathered = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)

    return gathered, log_probs  # [G, K], [G, K, vocab_size]
