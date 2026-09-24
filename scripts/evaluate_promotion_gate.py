"""Evaluate a V3 challenger against the frozen production champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relationscope.training.promotion_gate import PromotionGate, load_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("reports/open_set_final_report.json"),
    )
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = PromotionGate().compare(
        load_report(args.champion),
        load_report(args.challenger),
    )
    payload = result.model_dump(mode="json")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
