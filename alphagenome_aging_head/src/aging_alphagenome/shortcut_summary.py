"""Aggregate GC/composition audit and CNN stress-test results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev

STRATEGIES = ("random", "reference_only", "gc_tolerance", "gc_exact", "composition")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize composition-only and CNN shortcut stress-test metrics."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--aggregate-tsv", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict | None, *path: str):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_pattern = re.compile(r"^seed_(\d+)$")
    for seed_dir in sorted(root.expanduser().resolve().glob("seed_*")):
        match = seed_pattern.match(seed_dir.name)
        if not match:
            continue
        seed = int(match.group(1))
        for strategy in STRATEGIES:
            audit_path = seed_dir / f"audit_{strategy}" / "composition_audit.json"
            audit = _read_json(audit_path) if audit_path.is_file() else None
            original_metrics_path = seed_dir / f"cnn_{strategy}_original" / "metrics.json"
            shuffled_metrics_path = seed_dir / f"cnn_{strategy}_shuffled" / "metrics.json"
            original = _read_json(original_metrics_path) if original_metrics_path.is_file() else None
            shuffled = _read_json(shuffled_metrics_path) if shuffled_metrics_path.is_file() else None
            if audit is None and original is None and shuffled is None:
                continue

            gc_test = _metric(audit, "distribution_comparisons", "validation", "gc_fraction")
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "test_positive_gc_mean": _metric(gc_test, "positive", "mean"),
                    "test_negative_gc_mean": _metric(gc_test, "negative", "mean"),
                    "test_gc_mean_delta": _metric(gc_test, "positive_minus_negative_mean"),
                    "test_gc_cohen_d": _metric(gc_test, "cohen_d"),
                    "test_gc_ks": _metric(gc_test, "ks_statistic"),
                    "gc_only_auroc": _metric(audit, "composition_only_baselines", "gc_only", "test_metrics", "auroc"),
                    "mono_auroc": _metric(audit, "composition_only_baselines", "mononucleotide", "test_metrics", "auroc"),
                    "mono_dinuc_auroc": _metric(audit, "composition_only_baselines", "mono_plus_dinucleotide", "test_metrics", "auroc"),
                    "exact_gc_pair_fraction": _metric(audit, "paired_matching", "exact_gc_count_match_fraction"),
                    "cnn_original_accuracy": _metric(original, "metrics", "test", "accuracy"),
                    "cnn_original_auroc": _metric(original, "metrics", "test", "auroc"),
                    "cnn_original_ap": _metric(original, "metrics", "test", "average_precision"),
                    "cnn_shuffled_accuracy": _metric(shuffled, "metrics", "test", "accuracy"),
                    "cnn_shuffled_auroc": _metric(shuffled, "metrics", "test", "auroc"),
                    "cnn_shuffled_ap": _metric(shuffled, "metrics", "test", "average_precision"),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No seed_* stress-test results were found.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [name for name in rows[0] if name not in {"seed", "strategy"}]
    output: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        if not selected:
            continue
        record: dict[str, object] = {"strategy": strategy, "seeds": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            record[f"{metric}_mean"] = mean(values) if values else None
            record[f"{metric}_sd"] = stdev(values) if len(values) > 1 else (0.0 if values else None)
        output.append(record)
    return output


def main() -> int:
    args = parse_args()
    rows = collect(args.root)
    write_rows(args.output_tsv, rows)
    aggregate_path = args.aggregate_tsv or args.output_tsv.with_name(
        args.output_tsv.stem + ".aggregate.tsv"
    )
    write_rows(aggregate_path, aggregate(rows))
    print(f"Per-seed summary: {args.output_tsv.expanduser().resolve()}")
    print(f"Across-seed summary: {aggregate_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
