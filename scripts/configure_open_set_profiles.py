"""Attach named operating profiles to an open-set RelationScope bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROFILE_NAMES = ("high_coverage", "balanced", "high_safety")


def _threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--high-coverage", type=_threshold, required=True)
    parser.add_argument("--balanced", type=_threshold, required=True)
    parser.add_argument("--high-safety", type=_threshold, required=True)
    parser.add_argument(
        "--default-profile",
        choices=PROFILE_NAMES,
        default="balanced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.bundle_dir / "open_set_config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"open-set config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("open_set_config.json must contain a JSON object")

    profiles = {
        "high_coverage": {"threshold": args.high_coverage},
        "balanced": {"threshold": args.balanced},
        "high_safety": {"threshold": args.high_safety},
    }

    config["default_profile"] = args.default_profile
    config["profiles"] = profiles

    # Keep the historical key for backwards compatibility with older runtimes.
    config["abstention_threshold"] = profiles["balanced"]["threshold"]

    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
