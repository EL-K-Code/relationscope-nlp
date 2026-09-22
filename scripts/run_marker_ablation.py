"""Run controlled entity-information ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "A_raw_cls": ["--input-view", "raw", "--pooling-mode", "cls"],
    "B_marked_cls": ["--input-view", "marked", "--pooling-mode", "cls"],
    "C_marked_entity_pair": [
        "--input-view",
        "marked",
        "--pooling-mode",
        "entity_pair",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/ablations/entity_markers"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, object] = {
        "protocol": "entity_marker_ablation",
        "base_model": args.model_name,
        "conditions": {},
    }

    for name, condition_args in CONDITIONS.items():
        output_dir = args.output_root / name
        command = [
            sys.executable,
            "scripts/train_transformer.py",
            "--model-name",
            args.model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--output-dir",
            str(output_dir),
            *condition_args,
        ]
        print(f"\\n=== {name} ===")
        subprocess.run(command, check=True)

        metrics = json.loads(
            (output_dir / "metrics.json").read_text(encoding="utf-8")
        )
        summary["conditions"][name] = {
            "input_view": metrics["input_view"],
            "pooling_mode": metrics["pooling_mode"],
            "validation_macro_f1": metrics["best_validation_macro_f1"],
            "test": metrics["test"],
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "comparison.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
