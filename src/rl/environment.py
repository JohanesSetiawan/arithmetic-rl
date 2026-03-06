"""Rollout environment for GRPO.

For each question:
  1. Encode question as a prompt.
  2. Sample G completions stochastically (no_grad).
  3. Compute reward per completion.
  4. Compute group-relative advantages.
  5. Collect reference model log probs for KL penalty.
  6. Pack everything into an Experience ready for the GRPO update.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from ..model.transformer import ArithmeticTransformer
from ..tokenizer.tokenizer import ArithmeticTokenizer
from .buffer import Experience, pad_to_same_length


# ── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(predicted: Optional[int], ground_truth: int) -> float:
    """Two-level reward to bootstrap learning from a random-init model.

    1.0  — exact integer match   (accuracy reward)
    0.1  — valid integer, wrong  (format reward — encourages parseable output early on)
    0.0  — unparseable / gibberish

    Without the format reward, a random-init model has essentially zero probability
    of producing a correct answer, so advantages ≈ 0 and no learning occurs.
    This mirrors the format + accuracy reward strategy in DeepSeek-R1-Zero.
    """

    if predicted is None:
        return 0.0

    if predicted == ground_truth:
        return 1.0

    # Format reward HANYA jika jumlah digit sama dengan ground truth
    if len(str(abs(predicted))) == len(str(abs(ground_truth))):
        return 0.1

    return 0.0  # ← INI YANG DITAMBAH


def parse_answer(token_ids: list[int], tokenizer: ArithmeticTokenizer) -> Optional[int]:
    """Decode generated tokens → integer, or None if decoding fails."""
    try:
        return int(tokenizer.decode_answer(token_ids))
    except ValueError:
        return None


# ── Advantage ─────────────────────────────────────────────────────────────────

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-relative advantage normalisation.

    Â_i = (r_i − mean(r)) / (std(r) + ε)

    When all rewards in a group are identical, std ≈ 0 → advantages ≈ 0.
    This is the correct behaviour: no learning signal when all rollouts are equal.
    """
    return (returns - returns.mean()) / (returns.std() + eps)


# ── Log-prob helpers (no_grad) ────────────────────────────────────────────────

@torch.no_grad()
def generate_with_logprobs(
    model:          ArithmeticTransformer,
    prompt_ids:     torch.Tensor,   # [1, L]
    end_id:         int,
    max_new_tokens: int = 8,
    temperature:    float = 1.0,
) -> tuple[list[int], list[float]]:
    """Autoregressively sample one completion; return (token_ids, log_probs)."""
    model.eval()
    generated_ids: list[int] = []
    generated_lps: list[float] = []
    current = prompt_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(current)[0, -1].float()
        scaled_lps = F.log_softmax(logits / max(temperature, 1e-6), dim=-1)
        next_id = torch.multinomial(scaled_lps.exp(), num_samples=1).item()
        generated_ids.append(next_id)
        generated_lps.append(scaled_lps[next_id].item())
        if next_id == end_id:
            break
        current = torch.cat([current, torch.tensor(
            [[next_id]], device=current.device)], dim=1)

    return generated_ids, generated_lps


@torch.no_grad()
def compute_sequence_log_probs(
    model:      ArithmeticTransformer,
    sequence:   torch.Tensor,   # [T]  full sequence (prompt + answer)
    prompt_len: int,
) -> torch.Tensor:
    """Return per-token log probs of the answer portion under `model`.

    logits[:, i, :] predicts token at i+1, so answer tokens starting at
    prompt_len are predicted by logits at positions prompt_len-1 … T-2.

    Returns: [K]  where K = len(sequence) - prompt_len
    """
    model.eval()
    logits = model(sequence.unsqueeze(0)).float()           # [1, T, vocab]
    ans_logits = logits[0, prompt_len - 1: -1, :]             # [K, vocab]
    ans_ids = sequence[prompt_len:]                          # [K]
    log_probs = F.log_softmax(ans_logits, dim=-1)
    return log_probs.gather(-1, ans_ids.unsqueeze(-1)).squeeze(-1)  # [K]


# ── Main rollout function ─────────────────────────────────────────────────────

@torch.no_grad()
def rollout_question(
    model:          ArithmeticTransformer,
    ref_model:      ArithmeticTransformer,
    tokenizer:      ArithmeticTokenizer,
    question:       str,
    answer:         int,
    group_size:     int = 8,
    max_new_tokens: int = 8,
    temperature:    float = 1.0,
    device:         torch.device = torch.device("cpu"),
) -> Experience:
    """Sample `group_size` completions for one question; return packed Experience."""
    model.eval()
    ref_model.eval()

    prompt_ids = tokenizer.encode_question(question)
    prompt_tensor = torch.tensor(
        prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = len(prompt_ids)

    all_sequences: list[torch.Tensor] = []
    all_old_lps:   list[torch.Tensor] = []
    all_ref_lps:   list[torch.Tensor] = []
    all_rewards:   list[float] = []
    all_masks:     list[torch.Tensor] = []

    for _ in range(group_size):
        gen_ids, gen_lps = generate_with_logprobs(
            model, prompt_tensor, tokenizer.end_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        full_ids = prompt_ids + gen_ids
        seq_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
        old_lps = torch.tensor(gen_lps, dtype=torch.float32, device=device)
        mask = torch.ones(len(gen_ids), dtype=torch.bool, device=device)

        ref_lps = compute_sequence_log_probs(ref_model, seq_tensor, prompt_len)

        predicted = parse_answer(gen_ids, tokenizer)
        reward = compute_reward(predicted, answer)

        all_sequences.append(seq_tensor)
        all_old_lps.append(old_lps)
        all_ref_lps.append(ref_lps)
        all_rewards.append(reward)
        all_masks.append(mask)

    # Pad variable-length tensors to same length
    sequences_pad = pad_to_same_length(all_sequences, pad_value=0)
    old_lps_pad = pad_to_same_length(all_old_lps,   pad_value=0.0)
    ref_lps_pad = pad_to_same_length(all_ref_lps,   pad_value=0.0)
    mask_pad = pad_to_same_length(
        [m.float() for m in all_masks], pad_value=0.0
    ).bool()

    returns = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    advantages = group_advantages(returns)

    return Experience(
        sequences=sequences_pad,
        old_log_probs=old_lps_pad,
        ref_log_probs=ref_lps_pad,
        advantages=advantages,
        action_mask=mask_pad,
        returns=returns,
        prompt_len=prompt_len,
    )
