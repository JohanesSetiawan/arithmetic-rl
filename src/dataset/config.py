from dataclasses import dataclass, field
from pathlib import Path
from typing import List

SUPPORTED_OPERATIONS = ["addition", "subtraction", "multiplication", "division"]
SUPPORTED_FORMATS = ["jsonl", "csv"]


@dataclass
class DatasetConfig:
    num_samples: int
    output_path: Path
    operations: List[str] = field(default_factory=lambda: list(SUPPORTED_OPERATIONS))
    min_digits: int = 1
    max_digits: int = 3
    output_format: str = "jsonl"
    seed: int = 42

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not 1 <= self.min_digits <= self.max_digits <= 3:
            raise ValueError("digits must satisfy 1 <= min_digits <= max_digits <= 3")
        for op in self.operations:
            if op not in SUPPORTED_OPERATIONS:
                raise ValueError(f"Unsupported operation: {op}")
        if self.output_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {self.output_format}")
