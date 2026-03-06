import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import DatasetConfig

MAX_OPERAND = 999


@dataclass
class Sample:
    id: int
    question: str
    answer: int
    operation: str


class ArithmeticGenerator:
    """Generates arithmetic samples with balanced answer-digit distribution.

    Each sample targets a specific answer digit count first, then operands
    are chosen to land in that range. This prevents the severe bias in naive
    generators (e.g. 67% of division answers being 1 when operands are 3-digit).

    Operations available per target-digit tier:
        1-digit answers : division, subtraction, multiplication (small)
        2-digit answers : all four
        3-digit answers : addition, subtraction, division
    Multiplication is handled by explicitly factoring the target answer.
    """

    SYMBOLS: Dict[str, str] = {
        "addition": "+",
        "subtraction": "-",
        "multiplication": "*",
        "division": "/",
    }

    # Which operations can realistically produce answers in each digit tier
    _OPS_BY_TIER: Dict[int, List[str]] = {
        1: ["division", "subtraction", "multiplication"],
        2: ["addition", "subtraction", "division", "multiplication"],
        3: ["addition", "subtraction", "division"],
    }

    def __init__(self, config: DatasetConfig) -> None:
        config.validate()
        self.config = config
        self._rng = random.Random(config.seed)

    # ── Per-operation generators (answer-targeted) ────────────────────────────

    def _division(self, target_digits: int) -> Optional[Tuple[int, int, int]]:
        """Generate dividend/divisor = quotient with quotient in target range."""
        lo = 10 ** (target_digits - 1) if target_digits > 1 else 1
        hi = 10 ** target_digits - 1
        q = self._rng.randint(lo, hi)
        max_d = max(1, MAX_OPERAND // q)
        d = self._rng.randint(1, max_d)
        return d * q, d, q

    def _subtraction(self, target_digits: int) -> Optional[Tuple[int, int, int]]:
        """Generate a - b = answer with answer in target range."""
        lo = 10 ** (target_digits - 1) if target_digits > 1 else 1
        hi = 10 ** target_digits - 1
        ans = self._rng.randint(lo, hi)
        slack = MAX_OPERAND - ans
        if slack < 1:
            return None
        b = self._rng.randint(1, slack)
        a = ans + b
        return (a, b, ans) if a <= MAX_OPERAND else None

    def _addition(self, target_digits: int) -> Optional[Tuple[int, int, int]]:
        """Generate a + b = answer with answer in target range."""
        lo = 10 ** (target_digits - 1) if target_digits > 1 else 1
        hi = min(10 ** target_digits - 1, 2 * MAX_OPERAND)
        ans = self._rng.randint(lo, hi)
        a_lo = max(1, ans - MAX_OPERAND)
        a_hi = min(MAX_OPERAND, ans - 1)
        if a_lo > a_hi:
            return None
        a = self._rng.randint(a_lo, a_hi)
        b = ans - a
        return (a, b, ans) if 1 <= b <= MAX_OPERAND else None

    def _multiplication(self, target_digits: int) -> Optional[Tuple[int, int, int]]:
        """Generate a * b = answer with answer in target range.
        Samples answer first, then finds a valid factor pair by trial division."""
        lo = 10 ** (target_digits - 1) if target_digits > 1 else 1
        hi = 10 ** target_digits - 1
        ans = self._rng.randint(lo, hi)
        # Find all factors of ans that are ≤ MAX_OPERAND
        limit = min(MAX_OPERAND, int(ans ** 0.5) + 1)
        factors = [i for i in range(1, limit + 1) if ans % i == 0]
        if not factors:
            return None
        a = self._rng.choice(factors)
        b = ans // a
        return (a, b, ans) if 1 <= b <= MAX_OPERAND else None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> list:
        """Return a shuffled list of `num_samples` balanced arithmetic samples."""
        _generators = {
            "addition":       self._addition,
            "subtraction":    self._subtraction,
            "multiplication": self._multiplication,
            "division":       self._division,
        }

        requested_ops = set(self.config.operations)
        samples: list = []
        max_attempts = self.config.num_samples * 10

        for attempt in range(max_attempts):
            if len(samples) >= self.config.num_samples:
                break

            # Uniform target digit count → ensures each curriculum stage is balanced
            target_digits = self._rng.randint(
                self.config.min_digits, self.config.max_digits
            )

            # Pick a valid operation for this tier
            valid_ops = [
                op for op in self._OPS_BY_TIER.get(target_digits, list(requested_ops))
                if op in requested_ops
            ]
            if not valid_ops:
                continue
            op = valid_ops[len(samples) % len(valid_ops)]

            result = _generators[op](target_digits)
            if result is None:
                continue

            a, b, ans = result
            samples.append(Sample(
                id=len(samples) + 1,
                question=f"{a} {self.SYMBOLS[op]} {b} = ?",
                answer=ans,
                operation=op,
            ))

        if len(samples) < self.config.num_samples:
            raise RuntimeError(
                f"Generated only {len(samples)}/{self.config.num_samples} samples "
                f"after {max_attempts} attempts."
            )

        self._rng.shuffle(samples)
        for idx, s in enumerate(samples):
            s.id = idx + 1
        return samples
