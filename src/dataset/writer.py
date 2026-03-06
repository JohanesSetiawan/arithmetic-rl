import csv
import json
from pathlib import Path
from typing import Callable, Dict, List

from .generator import Sample


class DatasetWriter:
    """Serializes generated samples to JSONL or CSV."""

    def __init__(self, output_path: Path, output_format: str) -> None:
        self.output_path = output_path
        self.output_format = output_format

    def write(self, samples: List[Sample]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        writers: Dict[str, Callable] = {
            "jsonl": self._write_jsonl,
            "csv": self._write_csv,
        }
        writer_fn = writers.get(self.output_format)
        if writer_fn is None:
            raise ValueError(f"Unsupported format: {self.output_format}")
        writer_fn(samples)

    def _to_dict(self, s: Sample) -> dict:
        return {"id": s.id, "question": s.question, "answer": s.answer, "operation": s.operation}

    def _write_jsonl(self, samples: List[Sample]) -> None:
        with open(self.output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(self._to_dict(s)) + "\n")

    def _write_csv(self, samples: List[Sample]) -> None:
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "operation"])
            writer.writeheader()
            for s in samples:
                writer.writerow(self._to_dict(s))
