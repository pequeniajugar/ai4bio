"""Train and interpret a 201 bp DNA Transformer baseline with JAX.

The dataset contract intentionally follows ``aging_alphagenome.cnn``: training
uses only ``biological_sequence`` from the prepared TSV, keeps positive/control
pairs together for internal validation, and evaluates the supplied
chromosome-held-out validation split only after model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from aging_alphagenome.head import classification_metrics


PAD_ID = 0
BASE_TO_ID = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
}
ID_TO_BASE = {value: key for key, value in BASE_TO_ID.items()}
REVERSE_COMPLEMENT_IDS = np.asarray([0, 4, 3, 2, 1, 5], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a local-sequence Transformer on the prepared coordinate-only "
            "positive/random-control dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=201)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--ig-output",
        type=Path,
        help="Optional JSON file with Integrated Gradients region/motif analysis.",
    )
    parser.add_argument(
        "--ig-attributions-output",
        type=Path,
        help=(
            "Optional TSV with per-base Integrated Gradients rows for heatmaps. "
            "Columns include sample_id, position, base, and attribution."
        ),
    )
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--max-ig-examples", type=int, default=64)
    parser.add_argument("--ig-region-threshold-quantile", type=float, default=0.95)
    parser.add_argument(
        "--ig-split",
        choices=("train", "internal_validation", "test"),
        default="test",
        help="Dataset split to explain with Integrated Gradients.",
    )
    parser.add_argument("--motif-ngram-sizes", type=int, nargs="+", default=[3, 4, 5])
    return parser.parse_args()


def read_dataset(path: Path, sequence_length: int) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "sample_id",
            "pair_id",
            "chromosome",
            "position_1based",
            "label",
            "split",
            "biological_sequence",
        }
        if reader.fieldnames is None:
            raise ValueError("Prepared dataset has no header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Prepared dataset is missing columns: {sorted(missing)}"
            )
        rows.extend(reader)

    if not rows:
        raise ValueError("Prepared dataset contains no rows.")
    for row in rows:
        sequence = row["biological_sequence"].upper()
        if len(sequence) > sequence_length:
            raise ValueError(
                "Expected every biological_sequence to have length at most "
                f"{sequence_length}; {row['sample_id']} has length {len(sequence)}."
            )
        invalid = set(sequence) - set(BASE_TO_ID)
        if invalid:
            raise ValueError(
                f"{row['sample_id']} contains invalid bases: {sorted(invalid)}"
            )

    def strings(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=str)

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The Transformer dataset must contain labels 0 and 1.")

    sample_ids = [row["sample_id"] for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Every row must have a non-empty sample_id.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Every sample_id must be unique.")
    if any(not row["pair_id"] for row in rows):
        raise ValueError("Every row must have a non-empty pair_id.")
    observed_splits = {row["split"] for row in rows}
    if observed_splits != {"train", "validation"}:
        raise ValueError(
            "Expected exactly the train and validation splits; observed "
            f"{sorted(observed_splits)}."
        )

    pair_splits: dict[str, set[str]] = {}
    for row in rows:
        pair_splits.setdefault(row["pair_id"], set()).add(row["split"])
    crossing_pairs = sorted(
        pair_id for pair_id, splits in pair_splits.items() if len(splits) != 1
    )
    if crossing_pairs:
        raise ValueError(
            "pair_id values may not cross the train/validation boundary; "
            f"examples: {crossing_pairs[:5]}."
        )

    return {
        "sample_id": strings("sample_id"),
        "pair_id": strings("pair_id"),
        "chromosome": strings("chromosome"),
        "position_1based": np.asarray(
            [int(row["position_1based"]) for row in rows], dtype=np.int64
        ),
        "label": labels,
        "split": strings("split"),
        "sequence": np.asarray(
            [row["biological_sequence"].upper() for row in rows], dtype=str
        ),
    }


def make_development_masks(
    splits: np.ndarray,
    pair_ids: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("Internal validation fraction must be between 0 and 0.5.")
    development_mask = splits == "train"
    test_mask = splits == "validation"
    if not development_mask.any() or not test_mask.any():
        raise ValueError(
            "Prepared data must contain split=train and split=validation rows."
        )

    development_pairs = sorted(set(pair_ids[development_mask]))
    validation_pair_count = max(
        1, round(len(development_pairs) * validation_fraction)
    )

    def pair_hash(pair_id: str) -> str:
        return hashlib.sha256(f"{seed}|{pair_id}".encode()).hexdigest()

    internal_validation_pairs = set(
        sorted(development_pairs, key=pair_hash)[:validation_pair_count]
    )
    internal_validation_mask = development_mask & np.isin(
        pair_ids, list(internal_validation_pairs)
    )
    training_mask = development_mask & ~internal_validation_mask
    return training_mask, internal_validation_mask, test_mask


def tokenize_sequences(
    sequences: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((len(sequences), sequence_length), dtype=np.int32)
    mask = np.zeros((len(sequences), sequence_length), dtype=np.float32)
    for row_index, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        tokens[row_index, : len(sequence)] = [BASE_TO_ID[base] for base in sequence]
        mask[row_index, : len(sequence)] = 1.0
    return tokens, mask


def tokens_to_sequence(tokens: np.ndarray) -> str:
    return "".join(ID_TO_BASE.get(int(token), "") for token in tokens if token != PAD_ID)


def reverse_complement_tokens(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    complemented = REVERSE_COMPLEMENT_IDS[tokens]
    reversed_tokens = np.zeros_like(tokens)
    lengths = mask.astype(bool).sum(axis=1)
    for row_index, length in enumerate(lengths):
        if length:
            reversed_tokens[row_index, :length] = complemented[row_index, :length][::-1]
    return reversed_tokens


def reverse_complement_tokens_jax(jnp, tokens, mask):
    mapping = jnp.asarray(REVERSE_COMPLEMENT_IDS, dtype=tokens.dtype)
    complemented = mapping[tokens]
    sequence_length = tokens.shape[1]
    positions = jnp.arange(sequence_length)[None, :]
    lengths = jnp.sum(mask.astype(jnp.int32), axis=1, keepdims=True)
    source_positions = jnp.maximum(lengths - 1 - positions, 0)
    reversed_valid = jnp.take_along_axis(complemented, source_positions, axis=1)
    return jnp.where(positions < lengths, reversed_valid, PAD_ID)


def load_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def architecture_spec(sequence_length: int) -> dict[str, Any]:
    return {
        "name": "dna_transformer",
        "vocab_size": 6,
        "sequence_length": sequence_length,
        "d_model": 128,
        "position_embedding": "learnable",
        "encoder_layers": 4,
        "attention_heads": 4,
        "ffn_dim": 512,
        "dropout": 0.1,
        "pooling": "masked_mean",
        "classifier": (128, 64, 1),
        "loss": "BCEWithLogits",
        "optimizer": "AdamW",
        "initial_learning_rate": 1e-4,
    }


def architecture_parameter_count(sequence_length: int) -> int:
    spec = architecture_spec(sequence_length)
    d_model = spec["d_model"]
    ffn_dim = spec["ffn_dim"]
    layers = spec["encoder_layers"]
    total = spec["vocab_size"] * d_model
    total += sequence_length * d_model
    per_layer = 0
    per_layer += 4 * (d_model * d_model + d_model)
    per_layer += d_model * ffn_dim + ffn_dim
    per_layer += ffn_dim * d_model + d_model
    per_layer += 4 * d_model
    total += layers * per_layer
    total += d_model * 64 + 64
    total += 64 * 1 + 1
    return int(total)


def initialize_parameters(jax, jnp, key, sequence_length: int) -> dict[str, Any]:
    spec = architecture_spec(sequence_length)
    d_model = spec["d_model"]
    ffn_dim = spec["ffn_dim"]
    layer_count = spec["encoder_layers"]

    def normal(layer_key, shape, fan_in):
        return (
            jax.random.normal(layer_key, shape, dtype=jnp.float32)
            * np.sqrt(2.0 / fan_in)
        )

    names = ["token_embedding", "position_embedding"]
    for index in range(layer_count):
        prefix = f"encoder_{index}"
        names.extend(
            [
                f"{prefix}_q",
                f"{prefix}_k",
                f"{prefix}_v",
                f"{prefix}_attention_output",
                f"{prefix}_ffn1",
                f"{prefix}_ffn2",
            ]
        )
    names.extend(["classifier_hidden", "classifier_output"])
    split_keys = jax.random.split(key, len(names))
    keys = dict(zip(names, split_keys))

    parameters: dict[str, Any] = {
        "token_embedding": {
            "embedding": normal(keys["token_embedding"], (6, d_model), d_model)
        },
        "position_embedding": {
            "embedding": normal(
                keys["position_embedding"], (sequence_length, d_model), d_model
            )
        },
        "encoder": [],
        "classifier_hidden": {
            "weight": normal(keys["classifier_hidden"], (d_model, 64), d_model),
            "bias": jnp.zeros((64,), dtype=jnp.float32),
        },
        "classifier_output": {
            "weight": normal(keys["classifier_output"], (64, 1), 64),
            "bias": jnp.zeros((1,), dtype=jnp.float32),
        },
    }
    for index in range(layer_count):
        prefix = f"encoder_{index}"
        parameters["encoder"].append(
            {
                "attention_norm": {
                    "scale": jnp.ones((d_model,), dtype=jnp.float32),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "ffn_norm": {
                    "scale": jnp.ones((d_model,), dtype=jnp.float32),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "q": {
                    "weight": normal(keys[f"{prefix}_q"], (d_model, d_model), d_model),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "k": {
                    "weight": normal(keys[f"{prefix}_k"], (d_model, d_model), d_model),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "v": {
                    "weight": normal(keys[f"{prefix}_v"], (d_model, d_model), d_model),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "attention_output": {
                    "weight": normal(
                        keys[f"{prefix}_attention_output"],
                        (d_model, d_model),
                        d_model,
                    ),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
                "ffn1": {
                    "weight": normal(keys[f"{prefix}_ffn1"], (d_model, ffn_dim), d_model),
                    "bias": jnp.zeros((ffn_dim,), dtype=jnp.float32),
                },
                "ffn2": {
                    "weight": normal(keys[f"{prefix}_ffn2"], (ffn_dim, d_model), ffn_dim),
                    "bias": jnp.zeros((d_model,), dtype=jnp.float32),
                },
            }
        )
    return parameters


def layer_norm(jnp, values, parameters):
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    normalized = (values - mean) / jnp.sqrt(variance + 1e-5)
    return normalized * parameters["scale"] + parameters["bias"]


def apply_dropout(jax, jnp, values, key, rate: float, training: bool):
    if not training or rate <= 0.0:
        return values
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(key, keep_probability, shape=values.shape)
    return jnp.where(keep, values / keep_probability, 0.0)


def linear(values, parameters):
    return values @ parameters["weight"] + parameters["bias"]


def encoder_layer_apply(
    jax,
    jnp,
    parameters,
    values,
    mask,
    rng,
    dropout_rate: float,
    training: bool,
):
    d_model = values.shape[-1]
    head_count = 4
    head_dim = d_model // head_count
    key_iter = (
        iter(jax.random.split(rng, 3))
        if training and dropout_rate > 0.0
        else iter([None, None, None])
    )

    normalized = layer_norm(jnp, values, parameters["attention_norm"])
    q = linear(normalized, parameters["q"])
    k = linear(normalized, parameters["k"])
    v = linear(normalized, parameters["v"])
    batch_size, sequence_length, _ = q.shape
    q = q.reshape(batch_size, sequence_length, head_count, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, sequence_length, head_count, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, sequence_length, head_count, head_dim).transpose(0, 2, 1, 3)
    attention_logits = jnp.einsum("bhld,bhmd->bhlm", q, k) / jnp.sqrt(head_dim)
    key_mask = mask[:, None, None, :].astype(bool)
    attention_logits = jnp.where(key_mask, attention_logits, -1e9)
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    attention_weights = apply_dropout(
        jax, jnp, attention_weights, next(key_iter), dropout_rate, training
    )
    attended = jnp.einsum("bhlm,bhmd->bhld", attention_weights, v)
    attended = attended.transpose(0, 2, 1, 3).reshape(
        batch_size, sequence_length, d_model
    )
    attention_output = linear(attended, parameters["attention_output"])
    attention_output = apply_dropout(
        jax, jnp, attention_output, next(key_iter), dropout_rate, training
    )
    values = values + attention_output

    normalized = layer_norm(jnp, values, parameters["ffn_norm"])
    ffn = jax.nn.gelu(linear(normalized, parameters["ffn1"]))
    ffn = apply_dropout(jax, jnp, ffn, next(key_iter), dropout_rate, training)
    ffn = linear(ffn, parameters["ffn2"])
    ffn = apply_dropout(jax, jnp, ffn, None, dropout_rate, False)
    return values + ffn


def model_apply_from_embeddings(
    jax,
    jnp,
    parameters,
    token_embeddings,
    mask,
    rng,
    dropout_rate: float,
    training: bool,
):
    values = token_embeddings + parameters["position_embedding"]["embedding"][
        None, : token_embeddings.shape[1], :
    ]
    values = values * mask[:, :, None]
    key_iter = (
        iter(jax.random.split(rng, len(parameters["encoder"]) + 2))
        if training and dropout_rate > 0.0
        else iter([None] * (len(parameters["encoder"]) + 2))
    )
    for layer_parameters in parameters["encoder"]:
        values = encoder_layer_apply(
            jax,
            jnp,
            layer_parameters,
            values,
            mask,
            next(key_iter),
            dropout_rate,
            training,
        )
        values = values * mask[:, :, None]

    mask_sum = jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1.0)
    pooled = jnp.sum(values * mask[:, :, None], axis=1) / mask_sum
    hidden = jax.nn.gelu(linear(pooled, parameters["classifier_hidden"]))
    hidden = apply_dropout(jax, jnp, hidden, next(key_iter), dropout_rate, training)
    output = linear(hidden, parameters["classifier_output"])
    return output[:, 0]


def model_apply(
    jax,
    jnp,
    parameters,
    tokens,
    mask,
    rng,
    dropout_rate: float,
    training: bool,
):
    token_embeddings = parameters["token_embedding"]["embedding"][tokens]
    return model_apply_from_embeddings(
        jax, jnp, parameters, token_embeddings, mask, rng, dropout_rate, training
    )


def initialize_adam(jax, jnp, parameters):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)


def clip_gradients_by_global_norm(jax, jnp, gradients, max_norm: float = 1.0):
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive.")
    leaves = jax.tree_util.tree_leaves(gradients)
    gradients_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(gradient)) for gradient in leaves])
    )
    max_absolute_gradient = jnp.max(
        jnp.stack([jnp.max(jnp.abs(gradient)) for gradient in leaves])
    )
    safe_maximum = jnp.where(
        (max_absolute_gradient > 0.0) & jnp.isfinite(max_absolute_gradient),
        max_absolute_gradient,
        jnp.asarray(1.0, dtype=max_absolute_gradient.dtype),
    )
    scaled_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(gradient / safe_maximum)) for gradient in leaves)
    )
    normalized_clip_scale = max_norm / scaled_norm
    should_clip = (
        gradients_finite
        & (max_absolute_gradient > 0.0)
        & (max_absolute_gradient > max_norm / scaled_norm)
    )
    clipped = jax.tree_util.tree_map(
        lambda gradient: jnp.where(
            jnp.isfinite(gradient),
            jnp.where(
                should_clip,
                (gradient / safe_maximum) * normalized_clip_scale,
                gradient,
            ),
            0.0,
        ),
        gradients,
    )
    largest_representable = jnp.asarray(
        jnp.finfo(max_absolute_gradient.dtype).max,
        dtype=max_absolute_gradient.dtype,
    )
    representable_global_norm = jnp.where(
        max_absolute_gradient == 0.0,
        0.0,
        jnp.where(
            max_absolute_gradient <= largest_representable / scaled_norm,
            max_absolute_gradient * scaled_norm,
            largest_representable,
        ),
    )
    global_norm = jnp.where(
        gradients_finite,
        representable_global_norm,
        jnp.asarray(jnp.inf, dtype=max_absolute_gradient.dtype),
    )
    return clipped, global_norm, gradients_finite


def adamw_update(
    jax,
    jnp,
    parameters,
    gradients,
    optimizer_state,
    *,
    learning_rate: float,
    weight_decay: float,
):
    first_moment, second_moment, step = optimizer_state
    step = step + 1
    gradients, gradient_norm, gradients_finite = clip_gradients_by_global_norm(
        jax, jnp, gradients
    )
    beta1, beta2 = 0.9, 0.999
    first_moment = jax.tree_util.tree_map(
        lambda moment, gradient: beta1 * moment + (1.0 - beta1) * gradient,
        first_moment,
        gradients,
    )
    second_moment = jax.tree_util.tree_map(
        lambda moment, gradient: beta2 * moment
        + (1.0 - beta2) * gradient * gradient,
        second_moment,
        gradients,
    )
    first_scale = 1.0 - beta1**step
    second_scale = 1.0 - beta2**step
    updated = jax.tree_util.tree_map(
        lambda parameter, first, second: parameter
        - learning_rate
        * (
            (first / first_scale)
            / (jnp.sqrt(second / second_scale) + 1e-8)
            + weight_decay * parameter
        ),
        parameters,
        first_moment,
        second_moment,
    )
    return (
        updated,
        (first_moment, second_moment, step),
        gradient_norm,
        gradients_finite,
    )


def padded_batches(
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
):
    indices = np.asarray(indices, dtype=np.int64).copy()
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        valid_count = len(batch)
        if valid_count < batch_size:
            batch = np.concatenate(
                [batch, np.repeat(batch[-1:], batch_size - valid_count)]
            )
        mask = np.zeros(batch_size, dtype=np.float32)
        mask[:valid_count] = 1.0
        yield batch, mask


def make_predict_batch(jax, jnp, dropout_rate: float):
    @jax.jit
    def predict_batch(parameters, batch_tokens, batch_mask):
        forward_logits = model_apply(
            jax, jnp, parameters, batch_tokens, batch_mask, None, 0.0, False
        )
        reverse_tokens = reverse_complement_tokens_jax(jnp, batch_tokens, batch_mask)
        reverse_logits = model_apply(
            jax, jnp, parameters, reverse_tokens, batch_mask, None, 0.0, False
        )
        return (
            jax.nn.sigmoid(forward_logits) + jax.nn.sigmoid(reverse_logits)
        ) / 2.0

    return predict_batch


def predict_probabilities(
    predict_batch,
    jnp,
    parameters,
    tokens: np.ndarray,
    token_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for batch_indices, row_mask in padded_batches(indices, batch_size):
        probabilities = np.asarray(
            predict_batch(
                parameters,
                jnp.asarray(tokens[batch_indices]),
                jnp.asarray(token_mask[batch_indices]),
            ),
            dtype=np.float64,
        )
        predictions.append(probabilities[row_mask.astype(bool)])
    return np.concatenate(predictions)


def extended_classification_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(classification_metrics(labels, scores))
    labels_bool = labels.astype(bool)
    predictions = scores >= 0.5
    tp = int(np.sum(predictions & labels_bool))
    tn = int(np.sum(~predictions & ~labels_bool))
    fp = int(np.sum(predictions & ~labels_bool))
    fn = int(np.sum(~predictions & labels_bool))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    metrics.update(
        {
            "f1_at_0.5": float(f1),
            "pr_auc": metrics["average_precision"],
            "confusion_matrix_at_0.5": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            },
        }
    )
    return metrics


def flatten_parameters(parameters) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}

    def visit(prefix: str, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}{key}__", child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{prefix}{index}__", child)
        else:
            flattened[prefix[:-2]] = np.asarray(value)

    visit("", parameters)
    return flattened


def write_test_predictions(
    path: Path,
    dataset: dict[str, np.ndarray],
    test_indices: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "pair_id",
            "chromosome",
            "position_1based",
            "true_label",
            "predicted_probability",
            "predicted_label",
        ]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        if len(test_indices) != len(probabilities):
            raise ValueError("Test indices and probabilities have different lengths.")
        for index, probability in zip(test_indices, probabilities):
            writer.writerow(
                {
                    "sample_id": dataset["sample_id"][index],
                    "pair_id": dataset["pair_id"][index],
                    "chromosome": dataset["chromosome"][index],
                    "position_1based": int(dataset["position_1based"][index]),
                    "true_label": int(dataset["label"][index]),
                    "predicted_probability": f"{probability:.8f}",
                    "predicted_label": int(probability >= 0.5),
                }
            )
    temporary.replace(path)


def integrated_gradients_for_example(
    jax,
    jnp,
    parameters,
    tokens: np.ndarray,
    mask: np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    if steps < 1:
        raise ValueError("Integrated Gradients steps must be positive.")
    token_embedding = parameters["token_embedding"]["embedding"]
    input_embeddings = token_embedding[jnp.asarray(tokens[None, :])]
    baseline_tokens = jnp.zeros_like(jnp.asarray(tokens[None, :]))
    baseline_embeddings = token_embedding[baseline_tokens]
    batch_mask = jnp.asarray(mask[None, :])
    difference = input_embeddings - baseline_embeddings

    def logit_from_embeddings(candidate_embeddings):
        return model_apply_from_embeddings(
            jax,
            jnp,
            parameters,
            candidate_embeddings,
            batch_mask,
            None,
            0.0,
            False,
        )[0]

    gradient_fn = jax.grad(logit_from_embeddings)
    total_gradient = jnp.zeros_like(input_embeddings)
    for step in range(1, steps + 1):
        alpha = step / steps
        total_gradient = total_gradient + gradient_fn(
            baseline_embeddings + alpha * difference
        )
    attributions = jnp.sum(difference * (total_gradient / steps), axis=-1)[0]
    return np.asarray(attributions) * mask


def high_score_regions(
    attributions: np.ndarray,
    *,
    threshold_quantile: float,
) -> list[dict[str, Any]]:
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("IG region threshold quantile must be in (0, 1).")
    positive = np.maximum(attributions, 0.0)
    if not np.any(positive > 0.0):
        return []
    threshold = float(np.quantile(positive[positive > 0.0], threshold_quantile))
    active = positive >= threshold
    regions: list[dict[str, Any]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index < len(active) and active[index]:
            index += 1
        stop = index
        regions.append(
            {
                "start": int(start),
                "end": int(stop),
                "score": float(positive[start:stop].sum()),
            }
        )
    return regions


def ngram_counts(
    sequence: str,
    regions: list[dict[str, Any]],
    ngram_sizes: list[int],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {str(size): {} for size in ngram_sizes}
    for region in regions:
        subsequence = sequence[region["start"] : region["end"]]
        for size in ngram_sizes:
            if size <= 0:
                raise ValueError("Motif n-gram sizes must be positive.")
            for start in range(0, len(subsequence) - size + 1):
                ngram = subsequence[start : start + size]
                counts[str(size)][ngram] = counts[str(size)].get(ngram, 0) + 1
    return counts


def merge_ngram_counts(
    left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    merged = {size: dict(values) for size, values in left.items()}
    for size, values in right.items():
        merged.setdefault(size, {})
        for ngram, count in values.items():
            merged[size][ngram] = merged[size].get(ngram, 0) + count
    return merged


def write_integrated_gradients_report(
    path: Path,
    *,
    attributions_output: Optional[Path],
    jax,
    jnp,
    parameters,
    dataset: dict[str, np.ndarray],
    tokens: np.ndarray,
    token_mask: np.ndarray,
    probabilities: np.ndarray,
    explanation_indices: np.ndarray,
    split_name: str,
    max_examples: int,
    steps: int,
    threshold_quantile: float,
    ngram_sizes: list[int],
) -> None:
    ordered = explanation_indices[np.argsort(-probabilities)[:max_examples]]
    examples: list[dict[str, Any]] = []
    aggregate_counts: dict[str, dict[str, int]] = {str(size): {} for size in ngram_sizes}
    probability_by_index = {
        int(index): float(probability)
        for index, probability in zip(explanation_indices, probabilities)
    }
    attribution_rows: list[dict[str, Any]] = []
    for index in ordered:
        attributions = integrated_gradients_for_example(
            jax,
            jnp,
            parameters,
            tokens[index],
            token_mask[index],
            steps=steps,
        )
        regions = high_score_regions(
            attributions, threshold_quantile=threshold_quantile
        )
        sequence = tokens_to_sequence(tokens[index])
        probability = probability_by_index[int(index)]
        for position, (base, attribution) in enumerate(
            zip(sequence, attributions[: len(sequence)])
        ):
            attribution_rows.append(
                {
                    "sample_id": str(dataset["sample_id"][index]),
                    "label": int(dataset["label"][index]),
                    "predicted_probability": probability,
                    "position": position,
                    "relative_position": position - (len(sequence) // 2),
                    "base": base,
                    "attribution": float(attribution),
                }
            )
        counts = ngram_counts(sequence, regions, ngram_sizes)
        aggregate_counts = merge_ngram_counts(aggregate_counts, counts)
        examples.append(
            {
                "sample_id": str(dataset["sample_id"][index]),
                "label": int(dataset["label"][index]),
                "predicted_probability": probability,
                "regions": [
                    {
                        **region,
                        "sequence": sequence[region["start"] : region["end"]],
                    }
                    for region in regions
                ],
                "ngram_counts": counts,
            }
        )

    top_ngrams = {
        size: sorted(values.items(), key=lambda item: (-item[1], item[0]))[:25]
        for size, values in aggregate_counts.items()
    }
    payload = {
        "method": "Integrated Gradients on input token embeddings",
        "baseline": "PAD token embedding",
        "split": split_name,
        "steps": steps,
        "threshold_quantile": threshold_quantile,
        "max_examples": max_examples,
        "ngram_sizes": ngram_sizes,
        "attributions_output": str(attributions_output.expanduser().resolve())
        if attributions_output is not None
        else None,
        "top_ngrams": top_ngrams,
        "examples": examples,
    }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

    if attributions_output is not None:
        attributions_output = attributions_output.expanduser().resolve()
        attributions_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_attributions = attributions_output.with_suffix(
            attributions_output.suffix + ".tmp"
        )
        with temporary_attributions.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = [
                "sample_id",
                "label",
                "predicted_probability",
                "position",
                "relative_position",
                "base",
                "attribution",
            ]
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            for row in attribution_rows:
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "predicted_probability": (
                            f"{row['predicted_probability']:.8f}"
                        ),
                        "position": row["position"],
                        "relative_position": row["relative_position"],
                        "base": row["base"],
                        "attribution": f"{row['attribution']:.8g}",
                    }
                )
        temporary_attributions.replace(attributions_output)


def train(args: argparse.Namespace) -> int:
    if args.sequence_length < 1:
        raise ValueError("Sequence length must be positive.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("Dropout must be in [0, 1).")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("Learning rate must be finite and positive.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("Weight decay must be finite and non-negative.")
    if args.ig_attributions_output is not None and args.ig_output is None:
        raise ValueError(
            "--ig-attributions-output requires --ig-output so IG is computed."
        )

    dataset = read_dataset(args.dataset, args.sequence_length)
    training_mask, validation_mask, test_mask = make_development_masks(
        dataset["split"],
        dataset["pair_id"],
        validation_fraction=args.internal_validation_fraction,
        seed=args.seed,
    )
    labels = dataset["label"]
    for name, mask in (
        ("training", training_mask),
        ("internal validation", validation_mask),
        ("test", test_mask),
    ):
        if set(np.unique(labels[mask])) != {0, 1}:
            raise ValueError(f"{name} split must contain labels 0 and 1.")

    tokens, token_mask = tokenize_sequences(dataset["sequence"], args.sequence_length)
    training_indices = np.flatnonzero(training_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    jax, jnp = load_jax()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if args.require_gpu and not gpu_devices:
        raise RuntimeError("JAX exposes no GPU in this job.")
    device = gpu_devices[0] if gpu_devices else devices[0]
    print(f"Using {device}")
    print(
        f"Rows: train={len(training_indices):,}, "
        f"internal_validation={len(validation_indices):,}, "
        f"test={len(test_indices):,}"
    )
    print(
        "Architecture: dna_transformer "
        f"({architecture_parameter_count(args.sequence_length):,} parameters)"
    )

    key = jax.random.PRNGKey(args.seed)
    key, initialization_key = jax.random.split(key)
    with jax.default_device(device):
        parameters = initialize_parameters(
            jax, jnp, initialization_key, args.sequence_length
        )
        optimizer_state = initialize_adam(jax, jnp, parameters)

    @jax.jit
    def train_step(parameters, optimizer_state, batch_tokens, batch_mask, batch_y, row_mask, key):
        augmentation_key, dropout_key = jax.random.split(key)
        reverse_mask = jax.random.bernoulli(
            augmentation_key, 0.5, shape=(batch_tokens.shape[0], 1)
        )
        reverse_batch = reverse_complement_tokens_jax(jnp, batch_tokens, batch_mask)
        augmented = jnp.where(reverse_mask, reverse_batch, batch_tokens)

        def loss_function(candidate_parameters):
            logits = model_apply(
                jax,
                jnp,
                candidate_parameters,
                augmented,
                batch_mask,
                dropout_key,
                args.dropout,
                True,
            )
            losses = jnp.logaddexp(0.0, logits) - batch_y * logits
            loss = jnp.sum(losses * row_mask) / jnp.sum(row_mask)
            diagnostics = (
                jnp.max(jnp.abs(logits)),
                jnp.all(jnp.isfinite(logits)),
            )
            return loss, diagnostics

        (loss, (max_absolute_logit, logits_finite)), gradients = (
            jax.value_and_grad(loss_function, has_aux=True)(parameters)
        )
        parameters, optimizer_state, gradient_norm, gradients_finite = adamw_update(
            jax,
            jnp,
            parameters,
            gradients,
            optimizer_state,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        parameters_finite = jnp.all(
            jnp.stack(
                [
                    jnp.all(jnp.isfinite(value))
                    for value in jax.tree_util.tree_leaves(parameters)
                ]
            )
        )
        return (
            parameters,
            optimizer_state,
            loss,
            gradient_norm,
            max_absolute_logit,
            logits_finite,
            gradients_finite,
            parameters_finite,
        )

    predict_batch = make_predict_batch(jax, jnp, args.dropout)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_parameters = None
    epochs_without_improvement = 0
    started = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        batch_losses: list[float] = []
        batch_gradient_norms: list[float] = []
        batches = padded_batches(training_indices, args.batch_size, rng)
        for batch_number, (batch_indices, batch_row_mask) in enumerate(batches, start=1):
            key, step_key = jax.random.split(key)
            with jax.default_device(device):
                (
                    parameters,
                    optimizer_state,
                    loss,
                    gradient_norm,
                    max_absolute_logit,
                    logits_finite,
                    gradients_finite,
                    parameters_finite,
                ) = train_step(
                    parameters,
                    optimizer_state,
                    jnp.asarray(tokens[batch_indices]),
                    jnp.asarray(token_mask[batch_indices]),
                    jnp.asarray(labels[batch_indices], dtype=jnp.float32),
                    jnp.asarray(batch_row_mask),
                    step_key,
                )
            loss_value = float(loss)
            gradient_norm_value = float(gradient_norm)
            if (
                not np.isfinite(loss_value)
                or not bool(logits_finite)
                or not bool(gradients_finite)
                or not bool(parameters_finite)
            ):
                failing_ids = ", ".join(
                    str(value)
                    for value in dataset["sample_id"][batch_indices[:5]]
                )
                raise FloatingPointError(
                    "Non-finite Transformer state at "
                    f"epoch {epoch}, batch {batch_number}: loss={loss_value}, "
                    f"max_abs_logit={float(max_absolute_logit)}, "
                    f"gradient_norm={gradient_norm_value}, "
                    f"logits_finite={bool(logits_finite)}, "
                    f"gradients_finite={bool(gradients_finite)}, "
                    f"parameters_finite={bool(parameters_finite)}. "
                    f"First batch sample IDs: {failing_ids}."
                )
            batch_losses.append(loss_value)
            batch_gradient_norms.append(gradient_norm_value)

        validation_probabilities = predict_probabilities(
            predict_batch,
            jnp,
            parameters,
            tokens,
            token_mask,
            validation_indices,
            args.batch_size,
        )
        validation_metrics = extended_classification_metrics(
            labels[validation_indices], validation_probabilities
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_batch_loss": float(np.mean(batch_losses)),
            "training_mean_gradient_norm": float(np.mean(batch_gradient_norms)),
            "training_max_gradient_norm": float(np.max(batch_gradient_norms)),
            **{
                f"internal_validation_{name}": value
                for name, value in validation_metrics.items()
                if isinstance(value, float)
            },
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d}: train_loss={np.mean(batch_losses):.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_auroc={validation_metrics['auroc']:.4f} "
            f"val_f1={validation_metrics['f1_at_0.5']:.4f} "
            f"max_grad_norm={np.max(batch_gradient_norms):.3g}"
        )

        if validation_metrics["loss"] < best_validation_loss - 1e-5:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_parameters = jax.tree_util.tree_map(
                lambda value: np.asarray(value).copy(), parameters
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_parameters is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    parameters = jax.tree_util.tree_map(jnp.asarray, best_parameters)

    split_metrics: dict[str, dict[str, Any]] = {}
    split_probabilities: dict[str, np.ndarray] = {}
    for name, indices in (
        ("train", training_indices),
        ("internal_validation", validation_indices),
        ("test", test_indices),
    ):
        probabilities = predict_probabilities(
            predict_batch,
            jnp,
            parameters,
            tokens,
            token_mask,
            indices,
            args.batch_size,
        )
        split_probabilities[name] = probabilities
        split_metrics[name] = extended_classification_metrics(labels[indices], probabilities)

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output_model.with_suffix(output_model.suffix + ".tmp.npz")
    model_metadata = {
        "architecture": "dna_transformer",
        "specification": architecture_spec(args.sequence_length),
        "sequence_length": args.sequence_length,
        "dropout": args.dropout,
        "reverse_complement_augmentation": True,
        "reverse_complement_test_time_average": True,
        "best_epoch": best_epoch,
        "tokenization": {
            "PAD": PAD_ID,
            **BASE_TO_ID,
        },
    }
    np.savez_compressed(
        temporary_model,
        **flatten_parameters(best_parameters),
        metadata_json=np.asarray(json.dumps(model_metadata)),
    )
    temporary_model.replace(output_model)

    write_test_predictions(
        args.test_predictions,
        dataset,
        test_indices,
        split_probabilities["test"],
    )

    if args.ig_output is not None:
        write_integrated_gradients_report(
            args.ig_output,
            attributions_output=args.ig_attributions_output,
            jax=jax,
            jnp=jnp,
            parameters=parameters,
            dataset=dataset,
            tokens=tokens,
            token_mask=token_mask,
            probabilities=split_probabilities[args.ig_split],
            explanation_indices={
                "train": training_indices,
                "internal_validation": validation_indices,
                "test": test_indices,
            }[args.ig_split],
            split_name=args.ig_split,
            max_examples=args.max_ig_examples,
            steps=args.ig_steps,
            threshold_quantile=args.ig_region_threshold_quantile,
            ngram_sizes=args.motif_ngram_sizes,
        )

    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    parameter_count = sum(
        int(np.prod(value.shape))
        for value in jax.tree_util.tree_leaves(best_parameters)
    )
    expected_parameter_count = architecture_parameter_count(args.sequence_length)
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Expected {expected_parameter_count:,} parameters, "
            f"but initialized {parameter_count:,}."
        )
    payload = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "model": str(output_model),
        "test_predictions": str(args.test_predictions.expanduser().resolve()),
        "ig_output": str(args.ig_output.expanduser().resolve())
        if args.ig_output is not None
        else None,
        "ig_attributions_output": str(
            args.ig_attributions_output.expanduser().resolve()
        )
        if args.ig_attributions_output is not None
        else None,
        "device": str(device),
        "parameter_count": parameter_count,
        "counts": {
            "train": len(training_indices),
            "internal_validation": len(validation_indices),
            "test": len(test_indices),
        },
        "hyperparameters": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "internal_validation_fraction": args.internal_validation_fraction,
            "seed": args.seed,
        },
        "architecture": model_metadata,
        "metrics": split_metrics,
        "history": history,
        "test_policy": (
            "The supplied 20% chromosome-held-out split was evaluated only "
            "after model selection on an internal subset of the 80% split."
        ),
        "label_warning": (
            "Label 0 denotes a sampled random hg38 control, not an "
            "experimentally verified non-aging locus."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_metrics.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_metrics.replace(metrics_path)

    print(f"Best epoch: {best_epoch}")
    print(json.dumps({"test": split_metrics["test"]}, indent=2))
    print(f"Model:       {output_model}")
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {args.test_predictions.expanduser().resolve()}")
    if args.ig_output is not None:
        print(f"IG report:   {args.ig_output.expanduser().resolve()}")
    if args.ig_attributions_output is not None:
        print(
            "IG bases:    "
            f"{args.ig_attributions_output.expanduser().resolve()}"
        )
    return 0


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
