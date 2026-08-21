"""Audit GC/composition confounding in a prepared aging-locus dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from aging_alphagenome.composition import DINUCLEOTIDES, mononucleotide_shuffle, sequence_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure label/composition separation, train sequence-order-free "
            "baselines, and optionally write a composition-preserving shuffled dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-column", default="biological_sequence")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="validation")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--prediction-column",
        default="predicted_probability",
        help="Score column in --predictions; joined by sample_id.",
    )
    parser.add_argument(
        "--write-mononucleotide-shuffled",
        type=Path,
        help=(
            "Write a copy of the input TSV with the selected sequence column "
            "shuffled independently per row, preserving A/C/G/T/N counts exactly."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def _open_text(path: Path):
    path = path.expanduser().resolve()
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_rows(path: Path, sequence_column: str) -> tuple[list[dict[str, str]], list[str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Dataset has no header.")
        required = {"sample_id", "label", "split", sequence_column}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError("Dataset contains no rows.")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1; observed {sorted(labels)}")
    return rows, fieldnames


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    positive = labels == 1
    negative = labels == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order].astype(np.int8)
    positives = int(ordered.sum())
    if positives == 0:
        return float("nan")
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positives)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    predictions = probabilities >= 0.5
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "auroc": float(auroc(labels, probabilities)),
        "average_precision": float(average_precision(labels, probabilities)),
        "loss": float(
            -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        ),
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    l2: float,
    max_iterations: int,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    weights = np.zeros(train.shape[1], dtype=np.float64)
    regularizer = np.eye(train.shape[1], dtype=np.float64) * l2
    regularizer[0, 0] = 0.0
    n = len(train)
    for _ in range(max_iterations):
        probabilities = sigmoid(train @ weights)
        gradient = train.T @ (probabilities - train_y) / n + regularizer @ weights
        curvature = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        hessian = (train.T @ (train * curvature[:, None])) / n + regularizer
        hessian += np.eye(hessian.shape[0]) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        weights -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return sigmoid(test @ weights)


def cohen_d(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    pooled_numerator = (len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)
    pooled_denominator = len(first) + len(second) - 2
    pooled = np.sqrt(pooled_numerator / pooled_denominator) if pooled_denominator > 0 else 0.0
    if pooled < 1e-12:
        return 0.0 if np.isclose(first.mean(), second.mean()) else float("inf")
    return float((first.mean() - second.mean()) / pooled)


def ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([first, second])))
    if len(values) == 0:
        return float("nan")
    first_sorted = np.sort(first)
    second_sorted = np.sort(second)
    first_cdf = np.searchsorted(first_sorted, values, side="right") / len(first_sorted)
    second_cdf = np.searchsorted(second_sorted, values, side="right") / len(second_sorted)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 1001)
    distances = np.abs(
        np.quantile(first, quantiles) - np.quantile(second, quantiles)
    )
    widths = np.diff(quantiles)
    return float(np.sum((distances[:-1] + distances[1:]) * 0.5 * widths))


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def feature_matrix(feature_rows: list[dict[str, float | int]], names: Iterable[str]) -> np.ndarray:
    names = list(names)
    return np.asarray([[float(row[name]) for name in names] for row in feature_rows], dtype=np.float64)


def read_prediction_map(path: Path, score_column: str) -> dict[str, float]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames or score_column not in reader.fieldnames:
            raise ValueError("Prediction TSV must contain sample_id and the requested score column.")
        return {row["sample_id"]: float(row[score_column]) for row in reader}


def write_shuffled_dataset(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    sequence_column: str,
    seed: int,
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            copied[sequence_column] = mononucleotide_shuffle(
                row[sequence_column], seed=seed, key=row["sample_id"]
            )
            writer.writerow(copied)
    temporary.replace(path)


def json_safe(value):
    """Convert NumPy/non-finite scalars into strict JSON-compatible values."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    rows, fieldnames = read_rows(args.dataset, args.sequence_column)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = [sequence_features(row[args.sequence_column]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    splits = np.asarray([row["split"] for row in rows], dtype=str)

    feature_names = list(feature_rows[0].keys())
    feature_path = output_dir / "composition_features.tsv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        output_fields = ["sample_id", "pair_id", "chromosome", "label", "split"] + feature_names
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=output_fields, lineterminator="\n")
        writer.writeheader()
        for row, features in zip(rows, feature_rows, strict=True):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "pair_id": row.get("pair_id", ""),
                    "chromosome": row.get("chromosome", ""),
                    "label": row["label"],
                    "split": row["split"],
                    **features,
                }
            )

    audit_features = [
        "gc_fraction",
        "fraction_A",
        "fraction_C",
        "fraction_G",
        "fraction_T",
        "cpg_fraction",
        "shannon_entropy",
    ] + [f"dinuc_{dinuc}" for dinuc in DINUCLEOTIDES]
    comparisons: dict[str, dict[str, dict[str, float | int]]] = {}
    masks = {"all": np.ones(len(rows), dtype=bool)}
    for split in sorted(set(splits)):
        masks[split] = splits == split
    feature_arrays = {name: feature_matrix(feature_rows, [name])[:, 0] for name in audit_features}

    effects_rows: list[dict[str, object]] = []
    for split_name, mask in masks.items():
        positives = mask & (labels == 1)
        negatives = mask & (labels == 0)
        if not positives.any() or not negatives.any():
            continue
        comparisons[split_name] = {}
        local_labels = labels[mask]
        for name in audit_features:
            pos_values = feature_arrays[name][positives]
            neg_values = feature_arrays[name][negatives]
            raw_auc = auroc(local_labels, feature_arrays[name][mask])
            result = {
                "positive": summarize(pos_values),
                "negative": summarize(neg_values),
                "positive_minus_negative_mean": float(pos_values.mean() - neg_values.mean()),
                "cohen_d": cohen_d(pos_values, neg_values),
                "ks_statistic": ks_statistic(pos_values, neg_values),
                "wasserstein": wasserstein_1d(pos_values, neg_values),
                "univariate_auroc": float(raw_auc),
                "univariate_discriminability": float(max(raw_auc, 1.0 - raw_auc)),
            }
            comparisons[split_name][name] = result
            effects_rows.append({"split": split_name, "feature": name, **{k: v for k, v in result.items() if not isinstance(v, dict)}})

    effects_path = output_dir / "composition_effects.tsv"
    with effects_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "feature", "positive_minus_negative_mean", "cohen_d", "ks_statistic", "wasserstein", "univariate_auroc", "univariate_discriminability"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(effects_rows)

    histogram_path = output_dir / "gc_histogram.tsv"
    gc_edges = np.linspace(0.0, 1.0, 51)
    with histogram_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "label", "bin_start", "bin_end", "count"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split_name, mask in masks.items():
            for label in (0, 1):
                selected = mask & (labels == label)
                if not selected.any():
                    continue
                counts, _ = np.histogram(feature_arrays["gc_fraction"][selected], bins=gc_edges)
                for index, count in enumerate(counts):
                    writer.writerow({
                        "split": split_name,
                        "label": label,
                        "bin_start": f"{gc_edges[index]:.4f}",
                        "bin_end": f"{gc_edges[index + 1]:.4f}",
                        "count": int(count),
                    })

    train_mask = splits == args.train_split
    test_mask = splits == args.test_split
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Could not find both train split {args.train_split!r} and test split {args.test_split!r}."
        )
    if set(np.unique(labels[train_mask])) != {0, 1} or set(np.unique(labels[test_mask])) != {0, 1}:
        raise ValueError("Train and test splits must each contain both labels.")

    baseline_sets = {
        "gc_only": ["gc_fraction"],
        "mononucleotide": ["fraction_A", "fraction_C", "fraction_G", "fraction_T"],
        "mono_plus_cpg": ["fraction_A", "fraction_C", "fraction_G", "fraction_T", "cpg_fraction"],
        "mono_plus_dinucleotide": ["fraction_A", "fraction_C", "fraction_G", "fraction_T"] + [f"dinuc_{d}" for d in DINUCLEOTIDES],
    }
    baselines: dict[str, dict[str, object]] = {}
    for name, names in baseline_sets.items():
        matrix = feature_matrix(feature_rows, names)
        probabilities = fit_logistic_regression(
            matrix[train_mask],
            labels[train_mask].astype(np.float64),
            matrix[test_mask],
            l2=args.l2,
            max_iterations=args.max_iterations,
        )
        baselines[name] = {
            "features": names,
            "test_metrics": binary_metrics(labels[test_mask], probabilities),
        }

    pair_summary: dict[str, object] = {}
    if "pair_id" in fieldnames:
        by_pair: dict[str, dict[int, list[int]]] = {}
        for index, row in enumerate(rows):
            pair_id = row.get("pair_id", "")
            if pair_id:
                by_pair.setdefault(pair_id, {}).setdefault(int(row["label"]), []).append(index)
        gc_deltas: list[float] = []
        exact_gc_pairs = 0
        valid_pairs = 0
        for members in by_pair.values():
            if 1 not in members or 0 not in members:
                continue
            positive_index = members[1][0]
            for negative_index in members[0]:
                valid_pairs += 1
                delta = float(feature_arrays["gc_fraction"][positive_index] - feature_arrays["gc_fraction"][negative_index])
                gc_deltas.append(delta)
                if int(feature_rows[positive_index]["gc_count"]) == int(feature_rows[negative_index]["gc_count"]):
                    exact_gc_pairs += 1
        if valid_pairs:
            absolute = np.abs(np.asarray(gc_deltas, dtype=np.float64))
            pair_summary = {
                "paired_comparisons": valid_pairs,
                "exact_gc_count_matches": exact_gc_pairs,
                "exact_gc_count_match_fraction": exact_gc_pairs / valid_pairs,
                "mean_absolute_gc_fraction_delta": float(absolute.mean()),
                "max_absolute_gc_fraction_delta": float(absolute.max()),
            }

    prediction_analysis: dict[str, object] = {}
    if args.predictions:
        prediction_map = read_prediction_map(args.predictions, args.prediction_column)
        matched_indices = [index for index, row in enumerate(rows) if row["sample_id"] in prediction_map]
        if not matched_indices:
            raise ValueError("No prediction sample_id values matched the dataset.")
        scores = np.asarray([prediction_map[rows[index]["sample_id"]] for index in matched_indices], dtype=np.float64)
        gc = feature_arrays["gc_fraction"][matched_indices]
        correlation = float(np.corrcoef(gc, scores)[0, 1]) if len(scores) > 1 and gc.std() > 0 and scores.std() > 0 else float("nan")
        prediction_analysis = {
            "matched_rows": len(matched_indices),
            "pearson_score_vs_gc": correlation,
        }

    if args.write_mononucleotide_shuffled:
        write_shuffled_dataset(
            args.write_mononucleotide_shuffled,
            rows,
            fieldnames,
            args.sequence_column,
            args.seed,
        )

    report = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "sequence_column": args.sequence_column,
        "rows": len(rows),
        "split_counts": {
            split: {
                "positive": int(np.sum((splits == split) & (labels == 1))),
                "negative": int(np.sum((splits == split) & (labels == 0))),
            }
            for split in sorted(set(splits))
        },
        "distribution_comparisons": comparisons,
        "composition_only_baselines": baselines,
        "paired_matching": pair_summary,
        "prediction_analysis": prediction_analysis,
        "mononucleotide_shuffled_dataset": str(args.write_mononucleotide_shuffled.expanduser().resolve()) if args.write_mononucleotide_shuffled else None,
    }
    report_path = output_dir / "composition_audit.json"
    report_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    test_gc = comparisons.get(args.test_split, {}).get("gc_fraction", {})
    print(f"Audited {len(rows):,} rows from {args.dataset.expanduser().resolve()}")
    if test_gc:
        print(
            f"{args.test_split} GC mean: positive={test_gc['positive']['mean']:.4f}, "
            f"negative={test_gc['negative']['mean']:.4f}, "
            f"d={test_gc['cohen_d']:.3f}, KS={test_gc['ks_statistic']:.3f}, "
            f"GC-only AUROC={baselines['gc_only']['test_metrics']['auroc']:.4f}"
        )
    print("Composition-only held-out baselines:")
    for name, result in baselines.items():
        metrics = result["test_metrics"]
        print(f"  {name}: accuracy={metrics['accuracy']:.4f} auroc={metrics['auroc']:.4f} ap={metrics['average_precision']:.4f}")
    print(f"Report: {report_path}")
    print(f"Per-sample features: {feature_path}")
    print(f"Effect table: {effects_path}")
    print(f"GC histogram table: {histogram_path}")
    if args.write_mononucleotide_shuffled:
        print(f"Shuffled control dataset: {args.write_mononucleotide_shuffled.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
