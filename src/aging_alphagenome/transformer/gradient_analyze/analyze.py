"""Summarize and plot per-base Integrated Gradients attributions.

Input is the TSV written by ``aging_alphagenome.transformer.transformer`` with
``--ig-attributions-output``. The script intentionally depends only on the
standard library for tabular summaries; heatmap plotting uses matplotlib only
when requested.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "sample_id",
    "label",
    "predicted_probability",
    "position",
    "relative_position",
    "base",
    "attribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-base Integrated Gradients TSV output."
    )
    parser.add_argument("--attributions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of rows to keep per sample in sample_top_positions.tsv.",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        help="Optional PNG/PDF/SVG heatmap path. Requires matplotlib.",
    )
    parser.add_argument(
        "--heatmap-top-samples",
        type=int,
        default=50,
        help="Number of highest-probability samples to draw in the heatmap.",
    )
    parser.add_argument(
        "--label-filter",
        choices=("all", "0", "1"),
        default="all",
        help="Restrict summaries and heatmap to one label.",
    )
    return parser.parse_args()


def read_attributions(path: Path, label_filter: str) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Attribution TSV has no header.")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Attribution TSV is missing columns: {sorted(missing)}")
        for row in reader:
            label = int(row["label"])
            if label_filter != "all" and label != int(label_filter):
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "label": label,
                    "predicted_probability": float(row["predicted_probability"]),
                    "position": int(row["position"]),
                    "relative_position": int(row["relative_position"]),
                    "base": row["base"],
                    "attribution": float(row["attribution"]),
                }
            )
    if not rows:
        raise ValueError("No attribution rows remain after filtering.")
    return rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def summarize_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["position"], []).append(row)

    output: list[dict[str, Any]] = []
    for position, values in grouped.items():
        attributions = [row["attribution"] for row in values]
        base_counts: dict[str, int] = {}
        for row in values:
            base_counts[row["base"]] = base_counts.get(row["base"], 0) + 1
        top_base, top_base_count = sorted(
            base_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        output.append(
            {
                "position": position,
                "relative_position": values[0]["relative_position"],
                "count": len(values),
                "mean_attribution": sum(attributions) / len(attributions),
                "mean_abs_attribution": sum(abs(value) for value in attributions)
                / len(attributions),
                "positive_attribution_fraction": sum(
                    value > 0.0 for value in attributions
                )
                / len(attributions),
                "top_base": top_base,
                "top_base_count": top_base_count,
                "A_count": base_counts.get("A", 0),
                "C_count": base_counts.get("C", 0),
                "G_count": base_counts.get("G", 0),
                "T_count": base_counts.get("T", 0),
                "N_count": base_counts.get("N", 0),
            }
        )
    output.sort(key=lambda row: row["mean_abs_attribution"], reverse=True)
    return output


def summarize_positions_by_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["label"], row["position"]), []).append(row)

    output: list[dict[str, Any]] = []
    for (label, position), values in grouped.items():
        attributions = [row["attribution"] for row in values]
        output.append(
            {
                "label": label,
                "position": position,
                "relative_position": values[0]["relative_position"],
                "count": len(values),
                "mean_attribution": sum(attributions) / len(attributions),
                "mean_abs_attribution": sum(abs(value) for value in attributions)
                / len(attributions),
                "positive_attribution_fraction": sum(
                    value > 0.0 for value in attributions
                )
                / len(attributions),
            }
        )
    output.sort(
        key=lambda row: (row["label"], -row["mean_abs_attribution"], row["position"])
    )
    return output


def summarize_bases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["base"], []).append(row)

    output: list[dict[str, Any]] = []
    for base, values in grouped.items():
        attributions = [row["attribution"] for row in values]
        output.append(
            {
                "base": base,
                "count": len(values),
                "mean_attribution": sum(attributions) / len(attributions),
                "mean_abs_attribution": sum(abs(value) for value in attributions)
                / len(attributions),
                "positive_attribution_fraction": sum(
                    value > 0.0 for value in attributions
                )
                / len(attributions),
            }
        )
    output.sort(key=lambda row: row["mean_abs_attribution"], reverse=True)
    return output


def sample_top_positions(
    rows: list[dict[str, Any]], top_n: int
) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ValueError("--top-n must be positive.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], []).append(row)

    output: list[dict[str, Any]] = []
    for sample_id, values in grouped.items():
        values = sorted(
            values, key=lambda row: abs(row["attribution"]), reverse=True
        )[:top_n]
        for rank, row in enumerate(values, start=1):
            output.append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "label": row["label"],
                    "predicted_probability": row["predicted_probability"],
                    "position": row["position"],
                    "relative_position": row["relative_position"],
                    "base": row["base"],
                    "attribution": row["attribution"],
                    "abs_attribution": abs(row["attribution"]),
                }
            )
    output.sort(key=lambda row: (row["sample_id"], row["rank"]))
    return output


def write_heatmap(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    top_samples: int,
) -> None:
    if top_samples < 1:
        raise ValueError("--heatmap-top-samples must be positive.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Heatmap output requires matplotlib. Install it or omit --heatmap."
        ) from exc

    by_sample: dict[str, list[dict[str, Any]]] = {}
    probabilities: dict[str, float] = {}
    for row in rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
        probabilities[row["sample_id"]] = row["predicted_probability"]
    ordered_samples = sorted(
        by_sample, key=lambda sample_id: probabilities[sample_id], reverse=True
    )[:top_samples]
    positions = sorted({row["relative_position"] for row in rows})
    position_to_index = {position: index for index, position in enumerate(positions)}
    matrix = [
        [0.0 for _ in positions]
        for _ in ordered_samples
    ]
    for sample_index, sample_id in enumerate(ordered_samples):
        for row in by_sample[sample_id]:
            matrix[sample_index][position_to_index[row["relative_position"]]] = row[
                "attribution"
            ]

    maximum = max(abs(value) for sample in matrix for value in sample)
    maximum = maximum if maximum > 0.0 else 1.0
    width = max(10.0, min(24.0, len(positions) / 10.0))
    height = max(4.0, min(14.0, len(ordered_samples) / 4.0))
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
        interpolation="nearest",
    )
    tick_indices = list(range(0, len(positions), 10))
    axis.set_xticks(tick_indices)
    axis.set_xticklabels([str(positions[index]) for index in tick_indices], rotation=90)
    axis.set_yticks([])
    axis.set_xlabel("Position relative to center")
    axis.set_ylabel("Samples sorted by predicted probability")
    axis.set_title("Integrated Gradients per-base attribution")
    fig.colorbar(image, ax=axis, label="Attribution")
    fig.tight_layout()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_summary_json(
    path: Path,
    *,
    attributions: Path,
    rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    label_filter: str,
) -> None:
    sample_ids = {row["sample_id"] for row in rows}
    payload = {
        "attributions": str(attributions.expanduser().resolve()),
        "label_filter": label_filter,
        "row_count": len(rows),
        "sample_count": len(sample_ids),
        "top_positions_by_mean_abs_attribution": position_rows[:25],
        "base_summary": base_rows,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows = read_attributions(args.attributions, args.label_filter)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    position_rows = summarize_positions(rows)
    position_by_label_rows = summarize_positions_by_label(rows)
    base_rows = summarize_bases(rows)
    sample_rows = sample_top_positions(rows, args.top_n)

    write_tsv(
        output_dir / "position_importance.tsv",
        [
            "position",
            "relative_position",
            "count",
            "mean_attribution",
            "mean_abs_attribution",
            "positive_attribution_fraction",
            "top_base",
            "top_base_count",
            "A_count",
            "C_count",
            "G_count",
            "T_count",
            "N_count",
        ],
        position_rows,
    )
    write_tsv(
        output_dir / "position_importance_by_label.tsv",
        [
            "label",
            "position",
            "relative_position",
            "count",
            "mean_attribution",
            "mean_abs_attribution",
            "positive_attribution_fraction",
        ],
        position_by_label_rows,
    )
    write_tsv(
        output_dir / "base_importance.tsv",
        [
            "base",
            "count",
            "mean_attribution",
            "mean_abs_attribution",
            "positive_attribution_fraction",
        ],
        base_rows,
    )
    write_tsv(
        output_dir / "sample_top_positions.tsv",
        [
            "sample_id",
            "rank",
            "label",
            "predicted_probability",
            "position",
            "relative_position",
            "base",
            "attribution",
            "abs_attribution",
        ],
        sample_rows,
    )
    write_summary_json(
        output_dir / "summary.json",
        attributions=args.attributions,
        rows=rows,
        position_rows=position_rows,
        base_rows=base_rows,
        label_filter=args.label_filter,
    )
    if args.heatmap is not None:
        write_heatmap(rows, args.heatmap, top_samples=args.heatmap_top_samples)

    print(f"Rows:      {len(rows):,}")
    print(f"Samples:   {len({row['sample_id'] for row in rows}):,}")
    print(f"Output:    {output_dir}")
    if args.heatmap is not None:
        print(f"Heatmap:   {args.heatmap.expanduser().resolve()}")
    print("Top positions:")
    for row in position_rows[:10]:
        print(
            f"  pos={row['position']} rel={row['relative_position']:+d} "
            f"mean_abs={row['mean_abs_attribution']:.6g} "
            f"mean={row['mean_attribution']:.6g} base={row['top_base']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
