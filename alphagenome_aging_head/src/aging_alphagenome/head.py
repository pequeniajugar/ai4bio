"""Train and apply a linear aging-association head on frozen embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def binary_cross_entropy(
    labels: np.ndarray, logits: np.ndarray
) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    while index < len(scores):
        stop = index + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        average_rank = (index + 1 + stop) / 2.0
        ranks[order[index:stop]] = average_rank
        index = stop
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    cumulative_positives = np.cumsum(ordered_labels)
    precision = cumulative_positives / np.arange(1, len(labels) + 1)
    return float(precision[ordered_labels == 1].sum() / positives)


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be same-length 1D arrays.")
    if len(labels) == 0:
        raise ValueError("Cannot compute classification metrics on no rows.")
    if not np.isfinite(scores).all():
        raise ValueError("Classification scores contain NaN or infinity.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Classification scores must be probabilities in [0, 1].")
    predictions = scores >= 0.5
    labels_bool = labels.astype(bool)
    true_positive_rate = (
        float(predictions[labels_bool].mean())
        if labels_bool.any()
        else float("nan")
    )
    true_negative_rate = (
        float((~predictions[~labels_bool]).mean())
        if (~labels_bool).any()
        else float("nan")
    )
    return {
        "loss": binary_cross_entropy(
            labels.astype(np.float64),
            np.log(scores.clip(1e-12, 1 - 1e-12))
            - np.log1p(-scores.clip(1e-12, 1 - 1e-12)),
        ),
        "auroc": binary_auroc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "accuracy_at_0.5": float((predictions == labels_bool).mean()),
        "balanced_accuracy_at_0.5": (true_positive_rate + true_negative_rate)
        / 2.0,
    }


def fit_logistic_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    learning_rate: float,
    l2: float,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Fit a standardized linear sigmoid head with mini-batch Adam."""

    if train_features.ndim != 2:
        raise ValueError("Features must be a 2D matrix.")
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("Training labels must contain both 0 and 1.")
    if set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("Validation labels must contain both 0 and 1.")

    feature_mean = train_features.mean(axis=0, dtype=np.float64)
    feature_std = train_features.std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-6] = 1.0
    train_x = (
        (train_features - feature_mean) / feature_std
    ).astype(np.float32)
    validation_x = (
        (validation_features - feature_mean) / feature_std
    ).astype(np.float32)

    rng = np.random.default_rng(seed)
    weights = np.zeros(train_x.shape[1], dtype=np.float64)
    bias = 0.0
    first_moment_w = np.zeros_like(weights)
    second_moment_w = np.zeros_like(weights)
    first_moment_b = 0.0
    second_moment_b = 0.0
    adam_step = 0
    best_loss = float("inf")
    best_weights = weights.copy()
    best_bias = bias
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    positive_count = max(int(train_labels.sum()), 1)
    negative_count = max(len(train_labels) - positive_count, 1)
    class_weight = np.where(
        train_labels == 1,
        len(train_labels) / (2 * positive_count),
        len(train_labels) / (2 * negative_count),
    )

    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(train_x))
        for start in range(0, len(train_x), batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_labels[batch_indices].astype(np.float64)
            batch_weight = class_weight[batch_indices].astype(np.float64)
            logits = batch_x @ weights + bias
            probabilities = sigmoid(logits)
            residual = (probabilities - batch_y) * batch_weight
            normalization = batch_weight.sum()
            gradient_w = batch_x.T @ residual / normalization + l2 * weights
            gradient_b = float(residual.sum() / normalization)

            adam_step += 1
            first_moment_w = 0.9 * first_moment_w + 0.1 * gradient_w
            second_moment_w = (
                0.999 * second_moment_w + 0.001 * gradient_w * gradient_w
            )
            first_moment_b = 0.9 * first_moment_b + 0.1 * gradient_b
            second_moment_b = (
                0.999 * second_moment_b + 0.001 * gradient_b * gradient_b
            )
            corrected_moment_w = first_moment_w / (1 - 0.9**adam_step)
            corrected_second_w = second_moment_w / (1 - 0.999**adam_step)
            corrected_moment_b = first_moment_b / (1 - 0.9**adam_step)
            corrected_second_b = second_moment_b / (1 - 0.999**adam_step)
            weights -= learning_rate * corrected_moment_w / (
                np.sqrt(corrected_second_w) + 1e-8
            )
            bias -= learning_rate * corrected_moment_b / (
                np.sqrt(corrected_second_b) + 1e-8
            )

        train_logits = train_x @ weights + bias
        validation_logits = validation_x @ weights + bias
        train_loss = binary_cross_entropy(train_labels, train_logits)
        validation_loss = binary_cross_entropy(
            validation_labels, validation_logits
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_weights = weights.copy()
            best_bias = bias
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return (
        best_weights.astype(np.float32),
        float(best_bias),
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
        history,
    )


def load_feature_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as archive:
        required = {
            "features",
            "sample_id",
            "chromosome",
            "position_1based",
            "label",
            "split",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Feature archive is missing: {sorted(missing)}")
        return {name: archive[name] for name in archive.files}


def train_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def train_main() -> int:
    args = train_parse_args()
    archive = load_feature_archive(args.features)
    features = np.asarray(archive["features"], dtype=np.float32)
    labels = np.asarray(archive["label"], dtype=np.int8)
    splits = np.asarray(archive["split"], dtype=str)
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinity.")
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Both train and validation examples are required.")

    weights, bias, mean, std, history = fit_logistic_head(
        features[train_mask],
        labels[train_mask],
        features[validation_mask],
        labels[validation_mask],
        learning_rate=args.learning_rate,
        l2=args.l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    standardized = (features - mean) / std
    scores = sigmoid(standardized @ weights + bias)
    metrics = {
        "train": classification_metrics(labels[train_mask], scores[train_mask]),
        "validation": classification_metrics(
            labels[validation_mask], scores[validation_mask]
        ),
        "training": {
            "epochs_completed": len(history),
            "best_validation_loss": min(
                row["validation_loss"] for row in history
            ),
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        "label_warning": (
            "Label 0 is random/unlabeled hg38 sequence, not a proven "
            "non-aging locus."
        ),
    }

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_model.with_suffix(output_model.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        weights=weights,
        bias=np.asarray(bias, dtype=np.float32),
        feature_mean=mean,
        feature_std=std,
        threshold=np.asarray(0.5, dtype=np.float32),
    )
    temporary.replace(output_model)

    metrics_path = args.metrics or output_model.with_suffix(
        output_model.suffix + ".metrics.json"
    )
    metrics_path = metrics_path.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Head saved to {output_model}")
    return 0


def score_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def score_main() -> int:
    args = score_parse_args()
    archive = load_feature_archive(args.features)
    with np.load(args.model.expanduser().resolve(), allow_pickle=False) as model:
        weights = model["weights"]
        bias = float(model["bias"])
        mean = model["feature_mean"]
        std = model["feature_std"]
        threshold = float(model["threshold"])

    features = np.asarray(archive["features"], dtype=np.float32)
    if features.shape[1] != len(weights):
        raise ValueError(
            f"Feature dimension {features.shape[1]} does not match the head "
            f"dimension {len(weights)}."
        )
    scores = sigmoid(((features - mean) / std) @ weights + bias)
    output = args.output_tsv.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "sample_id",
                "chromosome",
                "position_1based",
                "aging_score",
                "predicted_label",
            ]
        )
        for index, score in enumerate(scores):
            writer.writerow(
                [
                    archive["sample_id"][index],
                    archive["chromosome"][index],
                    int(archive["position_1based"][index]),
                    f"{score:.8f}",
                    int(score >= threshold),
                ]
            )
    print(f"Scores saved to {output}")
    return 0
