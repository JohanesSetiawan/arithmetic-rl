"""3-stage curriculum learning manager.

Stages are defined by *answer* complexity (digit count of the result),
not operand complexity. This matters because the dataset has ~99 % 3-digit
operands, so operand-based staging would leave Stage 1 nearly empty.

Stage 1 → 2 → 3 advancement happens when validation accuracy ≥ threshold.
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ArithSample:
    question:   str
    answer:     int
    operation:  str
    ans_digits: int  # digit count of |answer| — used for curriculum staging


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ans_digits(answer: int) -> int:
    return len(str(abs(answer))) if answer != 0 else 1


def _load_jsonl(path: str) -> List[ArithSample]:
    samples = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            ans = int(obj["answer"])
            samples.append(ArithSample(
                question=obj["question"],
                answer=ans,
                operation=obj.get("operation", ""),
                ans_digits=_ans_digits(ans),
            ))
    return samples


# ── Manager ───────────────────────────────────────────────────────────────────

class CurriculumManager:
    """Dataset manager with stage-aware train/val sampling.

    Usage:
        mgr   = CurriculumManager("data/dataset.jsonl")
        batch = mgr.sample_train_batch(8)          # List[ArithSample]
        acc   = evaluate(model, mgr.val_pool)
        mgr.maybe_advance(acc)
    """

    def __init__(
        self,
        dataset_path:       str,
        val_fraction:       float = 0.10,
        held_out_fraction:  float = 0.10,
        seed:               int   = 42,
    ) -> None:
        self.rng   = random.Random(seed)
        self.stage = 1

        all_samples = _load_jsonl(dataset_path)
        self._build_splits(all_samples, val_fraction, held_out_fraction)

    # ── Data splits ───────────────────────────────────────────────────────────

    def _build_splits(
        self,
        all_samples: List[ArithSample],
        val_frac:    float,
        held_frac:   float,
    ) -> None:
        shuffled = list(all_samples)
        self.rng.shuffle(shuffled)

        n       = len(shuffled)
        n_val   = int(n * val_frac)
        n_held  = int(n * held_frac)

        train = shuffled[:n - n_val - n_held]
        val   = shuffled[n - n_val - n_held : n - n_held]
        self.held_out = shuffled[n - n_held:]

        # Bucket by answer digit count (cap at 3 for the 3 stages)
        self._train: Dict[int, List[ArithSample]] = {1: [], 2: [], 3: []}
        self._val:   Dict[int, List[ArithSample]] = {1: [], 2: [], 3: []}

        for s in train:
            self._train[min(s.ans_digits, 3)].append(s)
        for s in val:
            self._val[min(s.ans_digits, 3)].append(s)

    # ── Stage-aware pools ─────────────────────────────────────────────────────

    @property
    def train_pool(self) -> List[ArithSample]:
        """Training samples available at the current stage."""
        pool = []
        for d in range(1, self.stage + 1):
            pool.extend(self._train[d])
        return pool

    @property
    def val_pool(self) -> List[ArithSample]:
        """Validation samples for the current stage (tests hardest digits seen)."""
        pool = []
        for d in range(1, self.stage + 1):
            pool.extend(self._val[d])
        return pool

    # ── Sampling & advancement ────────────────────────────────────────────────

    def sample_train_batch(self, batch_size: int) -> List[ArithSample]:
        pool = self.train_pool
        if not pool:
            raise RuntimeError(f"Empty train pool at stage {self.stage}")
        return self.rng.choices(pool, k=batch_size)

    def maybe_advance(
        self,
        accuracy:      float,
        stage1_thresh: float = 0.70,
        stage2_thresh: float = 0.70,
    ) -> bool:
        """Advance to the next stage if accuracy meets the threshold.
        Returns True if the stage was advanced."""
        thresholds = {1: stage1_thresh, 2: stage2_thresh}
        if self.stage < 3 and accuracy >= thresholds.get(self.stage, 1.0):
            self.stage += 1
            print(f"  [Curriculum] → Stage {self.stage}  (acc={accuracy:.1%})")
            return True
        return False

    def info(self) -> str:
        return (
            f"Stage {self.stage} | "
            f"train={len(self.train_pool)} | "
            f"val={len(self.val_pool)}"
        )
