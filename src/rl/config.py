from dataclasses import dataclass


@dataclass
class RLConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset_path:    str = "data/dataset.jsonl"
    checkpoint_dir:  str = "checkpoints/rl"

    # ── GRPO core (DeepSeekMath paper) ────────────────────────────────────────
    group_size:  int = 16    # G: completions sampled per question
    clip_eps:    float = 0.2   # ε: PPO clipping range
    kl_weight:   float = 0.01  # β: KL penalty coefficient

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr:           float = 5e-5
    weight_decay: float = 0.01
    grad_clip:    float = 1.0

    # ── Training loop ─────────────────────────────────────────────────────────
    questions_per_step: int = 8     # questions per GRPO step
    total_steps:        int = 5000

    # ── Rollout ───────────────────────────────────────────────────────────────
    max_new_tokens: int = 8    # 6-digit answer + END = 7 tokens max, 8 is the cap
    temperature:    float = 1.0  # sampling temperature (1.0 = full entropy)

    # ── Curriculum ────────────────────────────────────────────────────────────
    stage1_acc_threshold: float = 0.65
    stage2_acc_threshold: float = 0.70

    # ── Evaluation & logging ──────────────────────────────────────────────────
    eval_interval: int = 50
    eval_samples:  int = 200
    log_interval:  int = 10
    save_interval: int = 200

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:   int = 42
    device: str = "cuda"  # or "cpu"
    entropy_weight: float = 0.0  # β_ent: entropy bonus coefficient
