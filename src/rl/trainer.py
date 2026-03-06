"""GRPO Trainer — orchestrates the full training loop.

One training step:
  1. Sample questions from curriculum.
  2. Rollout: generate G completions per question (no_grad).
  3. GRPO update: recompute current log probs (with grad) → clipped loss + KL.
  4. Backprop → gradient clip → optimizer step.
  5. Periodic eval + curriculum advancement + checkpointing.
"""

from __future__ import annotations
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_

from ..model.config import ModelConfig
from ..model.transformer import ArithmeticTransformer
from ..tokenizer.tokenizer import ArithmeticTokenizer
from .buffer import ReplayBuffer
from .config import RLConfig
from .curriculum import CurriculumManager
from .environment import parse_answer, rollout_question
from .grpo_loss import GRPOLoss
from .logprobs import compute_answer_log_probs_batch
from .wandb_logger import WandbLogger


class GRPOTrainer:
    """Pure RL arithmetic trainer using GRPO."""

    @staticmethod
    def _group_train_metrics(metrics: dict) -> dict[str, dict[str, float | int]]:
        return {
            "train/loss": {
                "total": metrics["loss"],
                "kl": metrics["kl"],
            },
            "train/reward": {
                "mean": metrics["mean_reward"],
                "std": metrics["reward_std"],
                "min": metrics["reward_min"],
                "max": metrics["reward_max"],
            },
            "train/system": {
                "grad_norm": metrics["grad_norm"],
                "learning_rate": metrics["learning_rate"],
                "step_time_sec": metrics["step_time_sec"],
            },
            "train/data": {
                "num_questions": metrics["num_questions"],
                "num_rollouts": metrics["num_rollouts"],
                "num_updates": metrics["num_updates"],
                "train_pool_size": metrics["train_pool_size"],
                "val_pool_size": metrics["val_pool_size"],
            },
            "train/curriculum": {
                "stage": metrics["stage"],
            },
        }

    @staticmethod
    def _group_eval_metrics(metrics: dict) -> dict[str, dict[str, float | int]]:
        return {
            "eval/accuracy": {
                "val_acc": metrics["val_acc"],
            },
            "eval/data": {
                "train_pool_size": metrics["train_pool_size"],
                "val_pool_size": metrics["val_pool_size"],
            },
            "eval/curriculum": {
                "stage": metrics["stage"],
                "stage_advanced": metrics["stage_advanced"],
            },
        }

    def __init__(self, model_config: ModelConfig, rl_config: RLConfig) -> None:
        self.cfg = rl_config
        self.device = torch.device(rl_config.device)

        torch.manual_seed(rl_config.seed)
        random.seed(rl_config.seed)

        self.tokenizer = ArithmeticTokenizer()

        # Policy model (trained) + frozen reference model (KL anchor)
        self.model = ArithmeticTransformer(model_config).to(self.device)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        self.objective = GRPOLoss(
            clip_eps=rl_config.clip_eps,
            kl_weight=rl_config.kl_weight,
            entropy_weight=rl_config.entropy_weight
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=rl_config.lr,
            weight_decay=rl_config.weight_decay,
        )

        self.curriculum = CurriculumManager(
            rl_config.dataset_path,
            val_fraction=0.10,
            held_out_fraction=0.10,
            seed=rl_config.seed,
        )
        wandb_config = asdict(rl_config)
        wandb_config.pop("wandb_token", None)
        self.wandb = WandbLogger(
            enabled=rl_config.wandb_enabled,
            project=rl_config.wandb_project,
            entity=rl_config.wandb_entity,
            run_name=rl_config.wandb_run_name,
            token=rl_config.wandb_token,
            config={
                "model": asdict(model_config),
                "rl": wandb_config,
            },
        )
        self.global_step = 0
        self._log:       list[dict] = []

        Path(rl_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(
            f"[GRPOTrainer] params={self.model.num_parameters():,}  device={self.device}")
        print(f"[GRPOTrainer] {self.curriculum.info()}")

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.cfg
        print(f"\n[GRPOTrainer] Training for {cfg.total_steps} steps\n")

        try:
            for step in range(1, cfg.total_steps + 1):
                self.global_step = step
                t0 = time.time()
                metrics = self._train_step()
                elapsed = time.time() - t0
                metrics["step_time_sec"] = elapsed

                if step % cfg.log_interval == 0:
                    print(
                        f"step {step:5d} | "
                        f"loss={metrics['loss']:.4f} | "
                        f"kl={metrics['kl']:.4f} | "
                        f"reward={metrics['mean_reward']:.3f} | "
                        f"stage={self.curriculum.stage} | "
                        f"{elapsed:.1f}s"
                    )

                self._log.append({"event": "train", "step": step, **metrics})
                self.wandb.log_groups(step, self._group_train_metrics(metrics))

                if step % cfg.eval_interval == 0:
                    val_acc = self._evaluate()
                    stage_advanced = self.curriculum.maybe_advance(
                        val_acc,
                        stage1_thresh=cfg.stage1_acc_threshold,
                        stage2_thresh=cfg.stage2_acc_threshold,
                    )
                    eval_metrics = {
                        "val_acc": val_acc,
                        "stage": self.curriculum.stage,
                        "stage_advanced": int(stage_advanced),
                        "val_pool_size": len(self.curriculum.val_pool),
                        "train_pool_size": len(self.curriculum.train_pool),
                    }
                    print(
                        f"  [eval] val_acc={val_acc:.1%}  {self.curriculum.info()}")
                    self._log.append(
                        {"event": "eval", "step": step, **eval_metrics})
                    self.wandb.log_groups(
                        step, self._group_eval_metrics(eval_metrics))

                if step % cfg.save_interval == 0:
                    self._save_checkpoint(step)

            print("\n[GRPOTrainer] Training complete.")
            self._save_checkpoint(self.global_step, name="final")
            self._save_log()
        finally:
            self.wandb.finish()

    # ── Training step ─────────────────────────────────────────────────────────

    def _train_step(self) -> dict:
        cfg = self.cfg
        questions = self.curriculum.sample_train_batch(cfg.questions_per_step)

        # Phase 1: rollout (no gradients)
        buffer = ReplayBuffer()
        all_rewards: list[float] = []

        with torch.no_grad():
            for sample in questions:
                exp = rollout_question(
                    model=self.model,
                    ref_model=self.ref_model,
                    tokenizer=self.tokenizer,
                    question=sample.question,
                    answer=sample.answer,
                    group_size=cfg.group_size,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    device=self.device,
                )
                buffer.append(exp)
                all_rewards.extend(exp.returns.tolist())

        # Phase 2: GRPO update (with gradients)
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, total_kl, n = 0.0, 0.0, 0
        grad_norm = 0.0

        for exp in buffer:
            exp = exp.to(self.device)

            current_lps, full_lps = compute_answer_log_probs_batch(
                self.model, exp.sequences, exp.prompt_len, exp.action_mask
            )
            loss, kl = self.objective(
                log_probs=current_lps,
                old_log_probs=exp.old_log_probs,
                ref_log_probs=exp.ref_log_probs,
                advantages=exp.advantages,
                action_mask=exp.action_mask,
                full_log_probs=full_lps,
            )

            if not loss.isfinite():
                print(
                    f"  [warn] non-finite loss at step {self.global_step} — skipping")
                self.optimizer.zero_grad()
                continue

            loss.backward()
            total_loss += loss.item()
            total_kl += kl.item()
            n += 1

        if n > 0:
            grad_norm = float(clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip).item())
            self.optimizer.step()

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        reward_min = min(all_rewards) if all_rewards else 0.0
        reward_max = max(all_rewards) if all_rewards else 0.0
        reward_var = sum((reward - mean_reward) **
                         2 for reward in all_rewards) / max(len(all_rewards), 1)
        n = max(n, 1)

        return {
            "loss":             total_loss / n,
            "kl":               total_kl / n,
            "mean_reward":      mean_reward,
            "reward_std":       reward_var ** 0.5,
            "reward_min":       reward_min,
            "reward_max":       reward_max,
            "stage":            self.curriculum.stage,
            "grad_norm":        grad_norm,
            "learning_rate":    self.optimizer.param_groups[0]["lr"],
            "num_questions":    len(questions),
            "num_rollouts":     len(all_rewards),
            "num_updates":      n,
            "train_pool_size":  len(self.curriculum.train_pool),
            "val_pool_size":    len(self.curriculum.val_pool),
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Greedy evaluation on the current-stage validation set."""
        self.model.eval()
        val_pool = self.curriculum.val_pool
        n_eval = len(val_pool)
        samples = random.sample(val_pool, n_eval)

        correct = sum(
            1 for s in samples
            if parse_answer(self._greedy_generate(s.question), self.tokenizer) == s.answer
        )
        return correct / n_eval if n_eval > 0 else 0.0

    def _greedy_generate(self, question: str) -> list[int]:
        """Greedy argmax generation for a single question."""
        prompt = torch.tensor(
            self.tokenizer.encode_question(question),
            dtype=torch.long, device=self.device
        ).unsqueeze(0)

        generated = []
        end_id = self.tokenizer.end_id

        for _ in range(self.cfg.max_new_tokens):
            next_id = self.model(prompt)[0, -1].argmax(-1).item()
            generated.append(next_id)
            if next_id == end_id:
                break
            prompt = torch.cat(
                [prompt, torch.tensor(
                    [[next_id]], dtype=torch.long, device=self.device)],
                dim=1
            )
        return generated

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, step: int, name: Optional[str] = None) -> None:
        fname = name or f"step_{step:05d}"
        path = Path(self.cfg.checkpoint_dir) / f"{fname}.pt"
        torch.save({
            "step":        step,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "stage":       self.curriculum.stage,
        }, path)
        print(f"  [ckpt] → {path}")

    def _save_log(self) -> None:
        path = Path(self.cfg.checkpoint_dir) / "training_log.json"
        with open(path, "w") as f:
            json.dump(self._log, f, indent=2)
        print(f"  [log]  → {path}")
