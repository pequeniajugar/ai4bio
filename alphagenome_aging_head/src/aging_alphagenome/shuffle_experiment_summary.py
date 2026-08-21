"""Summarize retraining and inference-only dinucleotide-shuffle experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_train_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    return payload["metrics"]["test"]


def _extract_score_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    return payload["metrics"]


def _row(
    variant: str,
    shuffle_seed: str,
    train_metrics: dict[str, float] | None,
    score_metrics: dict[str, float] | None,
) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "variant": variant,
        "shuffle_seed": shuffle_seed,
    }
    for prefix, metrics in (("retrained", train_metrics), ("fixed_model", score_metrics)):
        for metric in (
            "auroc",
            "average_precision",
            "accuracy_at_0.5",
            "balanced_accuracy_at_0.5",
            "loss",
        ):
            row[f"{prefix}_{metric}"] = (
                float(metrics[metric]) if metrics is not None else ""
            )
    return row


def _mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    rows: list[dict[str, str | float]] = []

    rows.append(
        _row(
            "original",
            "",
            _extract_train_metrics(root / "cnn_original" / "metrics.json"),
            _extract_score_metrics(root / "fixed_model_original" / "metrics.json"),
        )
    )
    rows.append(
        _row(
            "mono_shuffled",
            "17",
            _extract_train_metrics(root / "cnn_mono_seed_17" / "metrics.json"),
            _extract_score_metrics(root / "fixed_model_mono_seed_17" / "metrics.json"),
        )
    )

    dinuc_dirs = sorted(root.glob("cnn_dinuc_seed_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    for train_dir in dinuc_dirs:
        seed = train_dir.name.rsplit("_", 1)[1]
        rows.append(
            _row(
                "dinuc_shuffled",
                seed,
                _extract_train_metrics(train_dir / "metrics.json"),
                _extract_score_metrics(root / f"fixed_model_dinuc_seed_{seed}" / "metrics.json"),
            )
        )

    if not rows:
        raise ValueError(f"No experiment results found under {root}")

    output = args.output_tsv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate: dict[str, object] = {"rows": rows}
    dinuc_rows = [row for row in rows if row["variant"] == "dinuc_shuffled"]
    if dinuc_rows:
        aggregate["dinucleotide_shuffle"] = {}
        for metric in (
            "retrained_auroc",
            "retrained_average_precision",
            "retrained_accuracy_at_0.5",
            "fixed_model_auroc",
            "fixed_model_average_precision",
            "fixed_model_accuracy_at_0.5",
        ):
            values = [float(row[metric]) for row in dinuc_rows if row[metric] != ""]
            if values:
                mean, sd = _mean_sd(values)
                aggregate["dinucleotide_shuffle"][metric] = {
                    "n": len(values),
                    "mean": mean,
                    "sd": sd,
                }

    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Summary TSV:  {output}")
    print(f"Summary JSON: {json_path}")
    if "dinucleotide_shuffle" in aggregate:
        dinuc = aggregate["dinucleotide_shuffle"]
        if "retrained_auroc" in dinuc:
            stats = dinuc["retrained_auroc"]
            print(
                "Dinucleotide-shuffled retrained AUROC: "
                f"{stats['mean']:.4f} +/- {stats['sd']:.4f} (n={stats['n']})"
            )
        if "fixed_model_auroc" in dinuc:
            stats = dinuc["fixed_model_auroc"]
            print(
                "Dinucleotide-shuffled fixed-model AUROC: "
                f"{stats['mean']:.4f} +/- {stats['sd']:.4f} (n={stats['n']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
