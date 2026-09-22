"""Generate a model card from reproduced RelationScope artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--closed-set",
        type=Path,
        default=Path("artifacts/transformer/metrics.json"),
    )
    parser.add_argument(
        "--open-set",
        type=Path,
        default=Path("artifacts/open_set/metrics.json"),
    )
    parser.add_argument(
        "--latency",
        type=Path,
        default=Path("reports/latency.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/MODEL_CARD.md"),
    )
    args = parser.parse_args()

    closed = read_json(args.closed_set)
    open_set = read_json(args.open_set)
    latency = read_json(args.latency)

    lines = [
        "# RelationScope Model Card",
        "",
        (
            "> Generated only from reproduced artifact files. "
            "Missing experiments remain pending."
        ),
        "",
        "## Intended use",
        "",
        (
            "Relation classification between an explicitly supplied HEAD and TAIL "
            "entity in English FewRel-style sentences."
        ),
        "",
        (
            "The model is not a general-purpose fact checker and should not be "
            "treated as a source of truth outside its evaluated relation catalogue."
        ),
        "",
        "## Closed-set quality",
        "",
    ]

    if closed:
        test = closed.get("test", {})
        lines.extend(
            [
                f"- Protocol: {closed.get('protocol', 'unknown')}",
                f"- Base model: {closed.get('base_model', 'unknown')}",
                f"- Macro-F1: {fmt(test.get('macro_f1', 'n/a'))}",
                f"- Accuracy: {fmt(test.get('accuracy', 'n/a'))}",
                f"- Top-3 accuracy: {fmt(test.get('top3_accuracy', 'n/a'))}",
                f"- ECE: {fmt(test.get('ece_15', 'n/a'))}",
            ]
        )
    else:
        lines.append("- Pending reproduction.")

    lines.extend(["", "## Unseen-relation detection", ""])

    if open_set:
        ood = open_set.get("unseen_relation_test", {})
        lines.extend(
            [
                f"- Protocol: {open_set.get('protocol', 'unknown')}",
                f"- OOD AUROC: {fmt(ood.get('ood_auroc', 'n/a'))}",
                f"- OOD AUPRC: {fmt(ood.get('ood_auprc', 'n/a'))}",
                f"- FPR@95: {fmt(ood.get('ood_fpr95', 'n/a'))}",
                f"- Unknown F1: {fmt(ood.get('unknown_f1', 'n/a'))}",
                (
                    "- Acceptance threshold: "
                    f"{fmt(open_set.get('abstention_threshold', 'n/a'))}"
                ),
            ]
        )
    else:
        lines.append("- Pending reproduction.")

    lines.extend(["", "## Latency", ""])
    if latency:
        values = latency.get("latency_ms", {})
        lines.extend(
            [
                f"- Device: {latency.get('device', 'unknown')}",
                f"- p50: {fmt(values.get('median', 'n/a'), 1)} ms",
                f"- p95: {fmt(values.get('p95', 'n/a'), 1)} ms",
                f"- p99: {fmt(values.get('p99', 'n/a'), 1)} ms",
            ]
        )
    else:
        lines.append("- Pending benchmark.")

    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            (
                "- Entity spans are supplied explicitly; end-to-end NER is not "
                "hidden inside the relation classifier."
            ),
            (
                "- FewRel-style relation semantics do not guarantee transfer to "
                "every enterprise domain."
            ),
            (
                "- Open-set rejection is confidence-based and must be evaluated "
                "before being relied upon."
            ),
            "- Occlusion scores are evidence signals, not causal explanations.",
            "",
            "## Reproducibility",
            "",
            (
                "All metrics in this card must come from committed scripts and "
                "frozen protocol files."
            ),
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
