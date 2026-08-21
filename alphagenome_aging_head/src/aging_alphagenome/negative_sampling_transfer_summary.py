"""Summarize GC-matching versus GC-unmatched negative-sampling transfer tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_NAMES = (
    "auroc",
    "average_precision",
    "accuracy_at_0.5",
    "balanced_accuracy_at_0.5",
    "loss",
)
BASELINE_NAMES = (
    "gc_only",
    "mononucleotide",
    "mono_plus_cpg",
    "mono_plus_dinucleotide",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify paired positive examples and summarize the 2x2 CNN transfer "
            "matrix for GC-matched versus GC-unmatched negative sampling."
        )
    )
    parser.add_argument("--matched-dataset", type=Path, required=True)
    parser.add_argument("--unmatched-dataset", type=Path, required=True)
    parser.add_argument("--matched-audit", type=Path, required=True)
    parser.add_argument("--unmatched-audit", type=Path, required=True)
    parser.add_argument("--matched-on-matched", type=Path, required=True)
    parser.add_argument("--matched-on-unmatched", type=Path, required=True)
    parser.add_argument("--unmatched-on-matched", type=Path, required=True)
    parser.add_argument("--unmatched-on-unmatched", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dataset(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _positive_signature(rows: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    required = (
        "sample_id",
        "chromosome",
        "position_1based",
        "split",
        "biological_sequence",
    )
    positives: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if int(row["label"]) != 1:
            continue
        sample_id = row["sample_id"]
        positives[sample_id] = tuple(row[name] for name in required[1:])
    return positives


def verify_positive_identity(
    matched_rows: list[dict[str, str]],
    unmatched_rows: list[dict[str, str]],
) -> dict[str, Any]:
    matched = _positive_signature(matched_rows)
    unmatched = _positive_signature(unmatched_rows)
    missing_from_unmatched = sorted(set(matched) - set(unmatched))
    missing_from_matched = sorted(set(unmatched) - set(matched))
    mismatched = sorted(
        sample_id
        for sample_id in set(matched) & set(unmatched)
        if matched[sample_id] != unmatched[sample_id]
    )
    if missing_from_unmatched or missing_from_matched or mismatched:
        raise ValueError(
            "Positive examples differ between datasets. "
            f"missing_from_unmatched={missing_from_unmatched[:5]}, "
            f"missing_from_matched={missing_from_matched[:5]}, "
            f"mismatched={mismatched[:5]}"
        )
    return {
        "positive_count": len(matched),
        "identical_positive_ids_coordinates_splits_sequences": True,
    }


def _split_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split = row["split"]
        label = str(int(row["label"]))
        counts.setdefault(split, {"0": 0, "1": 0})[label] += 1
    return counts


def _extract_audit_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    comparisons = payload["distribution_comparisons"]
    test_split = "validation" if "validation" in comparisons else "test"
    gc = comparisons[test_split]["gc_fraction"]
    baselines = payload["composition_only_baselines"]
    row: dict[str, Any] = {
        "negative_regime": name,
        "test_split": test_split,
        "positive_mean_gc": float(gc["positive"]["mean"]),
        "negative_mean_gc": float(gc["negative"]["mean"]),
        "positive_minus_negative_gc": float(gc["positive_minus_negative_mean"]),
        "gc_cohen_d": float(gc["cohen_d"]),
        "gc_ks": float(gc["ks_statistic"]),
        "gc_univariate_auroc": float(gc["univariate_auroc"]),
    }
    for baseline in BASELINE_NAMES:
        row[f"{baseline}_auroc"] = float(
            baselines[baseline]["test_metrics"]["auroc"]
        )
        row[f"{baseline}_accuracy"] = float(
            baselines[baseline]["test_metrics"]["accuracy"]
        )
    paired = payload.get("paired_matching", {})
    row["mean_absolute_paired_gc_delta"] = paired.get(
        "mean_absolute_gc_fraction_delta", ""
    )
    row["max_absolute_paired_gc_delta"] = paired.get(
        "max_absolute_gc_fraction_delta", ""
    )
    return row


def _score_metrics(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = _read_json(path)
    metrics = payload["metrics"]
    # aging-score-cnn stores metrics directly; aging-train-cnn nests them by split.
    if "test" in metrics:
        metrics = metrics["test"]
    return {name: float(metrics[name]) for name in METRIC_NAMES}, payload


def _transfer_row(
    train_regime: str,
    test_regime: str,
    metrics_path: Path,
) -> dict[str, Any]:
    metrics, payload = _score_metrics(metrics_path)
    row: dict[str, Any] = {
        "train_negative_regime": train_regime,
        "test_negative_regime": test_regime,
        "dataset": str(payload.get("dataset", "")),
        "model": str(payload.get("model", "")),
    }
    row.update(metrics)
    return row


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_rows = _read_dataset(args.matched_dataset)
    unmatched_rows = _read_dataset(args.unmatched_dataset)
    positive_identity = verify_positive_identity(matched_rows, unmatched_rows)

    matched_counts = _split_counts(matched_rows)
    unmatched_counts = _split_counts(unmatched_rows)
    if matched_counts != unmatched_counts:
        raise ValueError(
            "Dataset split/label counts differ: "
            f"matched={matched_counts}, unmatched={unmatched_counts}"
        )

    audit_rows = [
        _extract_audit_row("gc_tolerance_matched", _read_json(args.matched_audit)),
        _extract_audit_row("gc_unmatched_reference_only", _read_json(args.unmatched_audit)),
    ]
    transfer_rows = [
        _transfer_row(
            "gc_tolerance_matched", "gc_tolerance_matched", args.matched_on_matched
        ),
        _transfer_row(
            "gc_tolerance_matched",
            "gc_unmatched_reference_only",
            args.matched_on_unmatched,
        ),
        _transfer_row(
            "gc_unmatched_reference_only",
            "gc_tolerance_matched",
            args.unmatched_on_matched,
        ),
        _transfer_row(
            "gc_unmatched_reference_only",
            "gc_unmatched_reference_only",
            args.unmatched_on_unmatched,
        ),
    ]

    composition_path = output_dir / "composition_comparison.tsv"
    transfer_path = output_dir / "transfer_matrix.tsv"
    _write_tsv(composition_path, audit_rows)
    _write_tsv(transfer_path, transfer_rows)

    by_pair = {
        (row["train_negative_regime"], row["test_negative_regime"]): row
        for row in transfer_rows
    }
    matched_native = by_pair[("gc_tolerance_matched", "gc_tolerance_matched")]
    unmatched_native = by_pair[
        ("gc_unmatched_reference_only", "gc_unmatched_reference_only")
    ]
    unmatched_to_matched = by_pair[
        ("gc_unmatched_reference_only", "gc_tolerance_matched")
    ]
    matched_to_unmatched = by_pair[
        ("gc_tolerance_matched", "gc_unmatched_reference_only")
    ]

    derived = {
        "unmatched_native_minus_matched_native_auroc": (
            unmatched_native["auroc"] - matched_native["auroc"]
        ),
        "unmatched_model_transfer_drop_auroc": (
            unmatched_native["auroc"] - unmatched_to_matched["auroc"]
        ),
        "matched_model_transfer_change_auroc": (
            matched_to_unmatched["auroc"] - matched_native["auroc"]
        ),
    }
    summary = {
        "design": {
            "matched": "same chromosome + center reference base + GC within +/-0.05",
            "gc_unmatched": (
                "same chromosome + center reference base, with no GC/composition matching"
            ),
            "reason_for_reference_only": (
                "Removing only GC matching isolates the GC-matching intervention; "
                "the repository's fully random mode would also remove center-base matching."
            ),
        },
        "positive_identity": positive_identity,
        "split_label_counts": matched_counts,
        "composition": audit_rows,
        "transfer_matrix": transfer_rows,
        "derived_auroc_differences": derived,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Verified {positive_identity['positive_count']:,} identical positives.")
    print("\nComposition comparison:")
    for row in audit_rows:
        print(
            f"  {row['negative_regime']}: "
            f"posGC={row['positive_mean_gc']:.4f} "
            f"negGC={row['negative_mean_gc']:.4f} "
            f"GC-AUROC={row['gc_only_auroc']:.4f} "
            f"dinuc-AUROC={row['mono_plus_dinucleotide_auroc']:.4f}"
        )
    print("\n2x2 CNN AUROC matrix:")
    print("  train\\test                      matched    GC-unmatched")
    print(
        "  GC-tolerance matched          "
        f"{matched_native['auroc']:.4f}     {matched_to_unmatched['auroc']:.4f}"
    )
    print(
        "  GC-unmatched reference-only   "
        f"{unmatched_to_matched['auroc']:.4f}     {unmatched_native['auroc']:.4f}"
    )
    print(f"\nComposition TSV: {composition_path}")
    print(f"Transfer TSV:    {transfer_path}")
    print(f"Summary JSON:    {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
