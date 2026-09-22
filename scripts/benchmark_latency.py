"""Benchmark local inference latency for a saved RelationScope bundle."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from relationscope.data.io import read_jsonl
from relationscope.data.schema import EntitySpan
from relationscope.inference.predictor import RelationPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/fewrel_closed_set_80/test.jsonl"),
    )
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/latency.json"),
    )
    return parser.parse_args()


def _span(text: str, mention: str) -> EntitySpan:
    start = text.find(mention)
    if start < 0:
        raise ValueError(f"mention not found: {mention!r}")
    return EntitySpan(text=mention, start=start, end=start + len(mention))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(int(round((len(ordered) - 1) * q)), len(ordered) - 1)
    return float(ordered[index])


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.dataset)[: args.examples]
    predictor = RelationPredictor(args.bundle, device=args.device)

    samples = []
    for record in records:
        text = " ".join(record.tokens)
        try:
            head = _span(text, record.head_text)
            tail = _span(text, record.tail_text)
        except ValueError:
            continue
        samples.append((text, head, tail))

    if not samples:
        raise RuntimeError("no benchmark examples had resolvable entity mentions")

    for text, head, tail in samples[: args.warmup]:
        predictor.predict(text=text, head=head, tail=tail, top_k=5)

    latencies: list[float] = []
    for text, head, tail in samples:
        started = time.perf_counter()
        predictor.predict(text=text, head=head, tail=tail, top_k=5)
        latencies.append((time.perf_counter() - started) * 1000)

    report = {
        "bundle": str(args.bundle),
        "device": str(predictor.device),
        "samples": len(latencies),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "min": min(latencies),
            "max": max(latencies),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
