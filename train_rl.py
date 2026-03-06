"""
GRPO training entry point.

The model learns arithmetic (1–3 digit operations) purely from reward signals.
No supervised answers are ever shown during training.

Usage:
    python train_rl.py                                  # CPU, 5 000 steps
    python train_rl.py --device cuda --steps 10000
    python train_rl.py --group-size 16 --lr 3e-5 --questions-per-step 16
"""

from src.rl.trainer import GRPOTrainer
from src.rl.config import RLConfig
from src.model.config import ModelConfig
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO arithmetic RL trainer")
    p.add_argument("--dataset",            default="data/dataset.jsonl")
    p.add_argument("--checkpoint-dir",     default="ckpt/rl")
    p.add_argument("--device",             default="cpu")
    p.add_argument("--steps",              type=int,   default=5000)
    p.add_argument("--lr",                 type=float, default=5e-5)
    p.add_argument("--group-size",         type=int,   default=8)
    p.add_argument("--clip-eps",           type=float, default=0.2)
    p.add_argument("--kl-weight",          type=float, default=0.01)
    p.add_argument("--temperature",        type=float, default=1.0)
    p.add_argument("--questions-per-step", type=int,   default=8)
    p.add_argument("--eval-interval",      type=int,   default=50)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--wandb-token",        default=None)
    p.add_argument("--wandb-project",      default=None)
    p.add_argument("--wandb-entity",       default=None)
    p.add_argument("--wandb-run-name",     default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wandb_enabled = any(
        value is not None
        for value in (
            args.wandb_token,
            args.wandb_project,
            args.wandb_entity,
            args.wandb_run_name,
        )
    )

    model_config = ModelConfig(
        vocab_size=20,
        d_model=256,
        n_heads=6,
        n_layers=4,
        ffn_dim=256,
        dropout=0.1,
    )

    rl_config = RLConfig(
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        total_steps=args.steps,
        lr=args.lr,
        group_size=args.group_size,
        clip_eps=args.clip_eps,
        kl_weight=args.kl_weight,
        temperature=args.temperature,
        questions_per_step=args.questions_per_step,
        eval_interval=args.eval_interval,
        seed=args.seed,
        wandb_enabled=wandb_enabled,
        wandb_project=args.wandb_project or (
            "arithmetic-rl" if wandb_enabled else None),
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_token=args.wandb_token,
    )

    GRPOTrainer(model_config, rl_config).train()


if __name__ == "__main__":
    main()
