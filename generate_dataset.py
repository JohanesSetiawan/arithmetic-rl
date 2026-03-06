"""
Dataset generator CLI.

Usage:
    python generate_dataset.py
    python generate_dataset.py --num-samples 10000 --output data/dataset.jsonl
    python generate_dataset.py --operations addition subtraction --max-digits 2
    python generate_dataset.py --format csv --output data/dataset.csv
"""

import argparse
from collections import Counter
from pathlib import Path

from src.dataset import ArithmeticGenerator, DatasetConfig, DatasetWriter
from src.dataset.config import SUPPORTED_FORMATS, SUPPORTED_OPERATIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate arithmetic dataset for RL training")
    p.add_argument("--num-samples",  type=int,   default=10000)
    p.add_argument("--output",       type=Path,  default=Path("data/dataset.jsonl"))
    p.add_argument("--format",       dest="output_format", choices=SUPPORTED_FORMATS, default="jsonl")
    p.add_argument("--operations",   nargs="+",  choices=SUPPORTED_OPERATIONS, default=SUPPORTED_OPERATIONS)
    p.add_argument("--min-digits",   type=int,   default=1)
    p.add_argument("--max-digits",   type=int,   default=3)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def print_stats(samples: list, path: Path) -> None:
    dist         = Counter(s.operation for s in samples)
    answer_range = (min(s.answer for s in samples), max(s.answer for s in samples))
    print(f"Saved    : {path}  ({len(samples)} samples)")
    print(f"Answers  : range [{answer_range[0]}, {answer_range[1]}]")
    print("Operations:")
    for op in SUPPORTED_OPERATIONS:
        if op in dist:
            print(f"  {op:<16} {dist[op]}")
    print("Preview:")
    for s in samples[:3]:
        print(f"  {s.question:<22} answer={s.answer}")


def main() -> None:
    args    = parse_args()
    config  = DatasetConfig(
        num_samples=args.num_samples,
        output_path=args.output,
        operations=args.operations,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        output_format=args.output_format,
        seed=args.seed,
    )
    samples = ArithmeticGenerator(config).generate()
    DatasetWriter(config.output_path, config.output_format).write(samples)
    print_stats(samples, config.output_path)


if __name__ == "__main__":
    main()
