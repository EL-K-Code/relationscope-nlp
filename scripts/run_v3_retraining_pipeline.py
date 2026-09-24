"""Run the V3 offline candidate retraining and promotion-gate pipeline.

This command deliberately stops before model registration/promotion. A human must
review the candidate report and gate result before any deployment action.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from relationscope.training.promotion_gate import PromotionGate, load_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v3_candidate"),
    )
    parser.add_argument(
        "--champion-report",
        type=Path,
        default=Path("reports/open_set_final_report.json"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_command = [
        sys.executable,
        "scripts/train_open_set_transformer.py",
        "--review-snapshot",
        str(args.review_snapshot),
        "--candidate-model-version",
        args.candidate_model_version,
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
    ]
    print("Running candidate training:")
    print(" ".join(train_command))
    subprocess.run(train_command, check=True)

    challenger_report = args.output_dir / "candidate_report.json"
    result = PromotionGate().compare(
        load_report(args.champion_report),
        load_report(challenger_report),
    )
    gate_path = args.output_dir / "promotion_gate.json"
    gate_path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "candidate_model_version": args.candidate_model_version,
                "candidate_report": str(challenger_report),
                "promotion_gate": str(gate_path),
                "gate_passed": result.passed,
                "next_action": (
                    "review_then_register_challenger"
                    if result.passed
                    else "do_not_promote"
                ),
            },
            indent=2,
        )
    )

    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
