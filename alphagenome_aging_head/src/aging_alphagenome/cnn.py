"""Train and evaluate a compact 201 bp DNA CNN baseline with JAX."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from aging_alphagenome.features import one_hot_encode
from aging_alphagenome.head import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a local-sequence CNN on the prepared coordinate-only "
            "positive/random-control dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        choices=("small", "large"),
        default="small",
        help="CNN capacity preset; small reproduces the original baseline.",
    )
    parser.add_argument("--sequence-length", type=int, default=201)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--internal-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless JAX exposes a GPU (recommended for the Slurm job).",
    )
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
    lengths = {len(row["biological_sequence"]) for row in rows}
    if lengths != {sequence_length}:
        raise ValueError(
            "Expected every biological_sequence to have length "
            f"{sequence_length}; observed {sorted(lengths)}."
        )
    for row in rows:
        invalid = set(row["biological_sequence"].upper()) - set("ACGTN")
        if invalid:
            raise ValueError(
                f"{row['sample_id']} contains invalid bases: {sorted(invalid)}"
            )

    def strings(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=str)

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The CNN dataset must contain labels 0 and 1.")

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
    """Split training pairs for early stopping and preserve held-out test rows."""

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


def reverse_complement(encoded: np.ndarray) -> np.ndarray:
    """Reverse-complement one-hot A/C/G/T sequences."""

    return encoded[:, ::-1, :][:, :, [3, 2, 1, 0]]


def load_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def architecture_spec(name: str) -> dict[str, Any]:
    """Return a serializable description of a supported CNN."""

    if name == "small":
        return {
            "name": "small",
            "conv_layers": (
                {
                    "name": "conv1",
                    "kernel": 15,
                    "input_channels": 4,
                    "output_channels": 64,
                    "dilation": 1,
                },
                {
                    "name": "conv2",
                    "kernel": 7,
                    "input_channels": 64,
                    "output_channels": 96,
                    "dilation": 1,
                },
                {
                    "name": "conv3",
                    "kernel": 5,
                    "input_channels": 96,
                    "output_channels": 128,
                    "dilation": 1,
                },
            ),
            "pool_after": ("conv1", "conv2"),
            "dense_layers": (
                {
                    "name": "dense",
                    "input_features": 256,
                    "output_features": 128,
                },
            ),
            "receptive_field_bp": 46,
        }
    if name == "large":
        return {
            "name": "large",
            "conv_layers": (
                {
                    "name": "conv1",
                    "kernel": 15,
                    "input_channels": 4,
                    "output_channels": 128,
                    "dilation": 1,
                },
                {
                    "name": "conv2",
                    "kernel": 9,
                    "input_channels": 128,
                    "output_channels": 256,
                    "dilation": 1,
                },
                {
                    "name": "conv3",
                    "kernel": 7,
                    "input_channels": 256,
                    "output_channels": 256,
                    "dilation": 1,
                },
                {
                    "name": "conv4",
                    "kernel": 7,
                    "input_channels": 256,
                    "output_channels": 384,
                    "dilation": 2,
                },
                {
                    "name": "conv5",
                    "kernel": 5,
                    "input_channels": 384,
                    "output_channels": 384,
                    "dilation": 2,
                },
                {
                    "name": "conv6",
                    "kernel": 5,
                    "input_channels": 384,
                    "output_channels": 512,
                    "dilation": 4,
                },
            ),
            "pool_after": ("conv1", "conv3"),
            "dense_layers": (
                {
                    "name": "dense1",
                    "input_features": 1024,
                    "output_features": 512,
                },
                {
                    "name": "dense2",
                    "input_features": 512,
                    "output_features": 128,
                },
            ),
            "receptive_field_bp": 190,
        }
    raise ValueError(f"Unknown architecture: {name}")


def architecture_parameter_count(name: str) -> int:
    spec = architecture_spec(name)
    total = 0
    for layer in spec["conv_layers"]:
        total += (
            layer["kernel"]
            * layer["input_channels"]
            * layer["output_channels"]
            + layer["output_channels"]
        )
    for layer in spec["dense_layers"]:
        total += (
            layer["input_features"] * layer["output_features"]
            + layer["output_features"]
        )
    final_features = spec["dense_layers"][-1]["output_features"]
    return total + final_features + 1


def initialize_parameters(
    jax, jnp, key, architecture: str
) -> dict[str, dict[str, Any]]:
    """Initialize one of the supported convolutional architectures."""

    spec = architecture_spec(architecture)
    layer_specs: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for layer in spec["conv_layers"]:
        layer_specs[layer["name"]] = (
            (
                layer["kernel"],
                layer["input_channels"],
                layer["output_channels"],
            ),
            (layer["output_channels"],),
        )
    for layer in spec["dense_layers"]:
        layer_specs[layer["name"]] = (
            (layer["input_features"], layer["output_features"]),
            (layer["output_features"],),
        )
    final_features = spec["dense_layers"][-1]["output_features"]
    layer_specs["output"] = ((final_features, 1), (1,))

    keys = iter(jax.random.split(key, len(layer_specs)))
    parameters: dict[str, dict[str, Any]] = {}
    for name, (weight_shape, bias_shape) in layer_specs.items():
        layer_key = next(keys)
        fan_in = int(np.prod(weight_shape[:-1]))
        weights = (
            jax.random.normal(layer_key, weight_shape, dtype=jnp.float32)
            * np.sqrt(2.0 / fan_in)
        )
        parameters[name] = {
            "weight": weights,
            "bias": jnp.zeros(bias_shape, dtype=jnp.float32),
        }
    return parameters


def model_apply(
    jax,
    jnp,
    parameters,
    sequences,
    rng,
    dropout_rate,
    training,
    architecture,
):
    spec = architecture_spec(architecture)

    def convolution(values, layer_spec):
        layer = parameters[layer_spec["name"]]
        return jax.lax.conv_general_dilated(
            values,
            layer["weight"],
            window_strides=(1,),
            padding="SAME",
            rhs_dilation=(layer_spec["dilation"],),
            dimension_numbers=("NWC", "WIO", "NWC"),
        ) + layer["bias"]

    def max_pool(values):
        return jax.lax.reduce_window(
            values,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 2, 1),
            window_strides=(1, 2, 1),
            padding="VALID",
        )

    values = sequences
    pool_after = set(spec["pool_after"])
    for specification in spec["conv_layers"]:
        values = jax.nn.gelu(convolution(values, specification))
        if specification["name"] in pool_after:
            values = max_pool(values)
    values = jnp.concatenate(
        [jnp.mean(values, axis=1), jnp.max(values, axis=1)], axis=-1
    )
    dropout_keys = (
        iter(jax.random.split(rng, len(spec["dense_layers"])))
        if training and dropout_rate
        else None
    )
    for specification in spec["dense_layers"]:
        dense = parameters[specification["name"]]
        values = jax.nn.gelu(values @ dense["weight"] + dense["bias"])
        if dropout_keys is not None:
            keep_probability = 1.0 - dropout_rate
            keep = jax.random.bernoulli(
                next(dropout_keys), keep_probability, shape=values.shape
            )
            values = jnp.where(keep, values / keep_probability, 0.0)
    output = parameters["output"]
    return (values @ output["weight"] + output["bias"])[:, 0]


def initialize_adam(jax, jnp, parameters):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return zeros, zeros, jnp.asarray(0, dtype=jnp.int32)


def clip_gradients_by_global_norm(jax, jnp, gradients, max_norm: float = 1.0):
    """Clip finite gradients without overflowing while computing their norm.

    Squaring an otherwise finite float32 gradient larger than roughly 1e19 can
    overflow. The previous direct sum-of-squares implementation then produced
    an infinite norm and silently scaled every finite gradient to zero. If a
    raw gradient was already infinite, the same code evaluated ``inf * 0`` and
    poisoned the optimizer state and parameters with NaNs. Scaling before the
    sum avoids the first case; the returned finite flag lets training stop at
    the originating batch in the second case.
    """

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
        sum(
            jnp.sum(jnp.square(gradient / safe_maximum))
            for gradient in leaves
        )
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


def make_predict_batch(jax, jnp, architecture):
    """Build the inference function once so JAX compiles it once per shape."""

    @jax.jit
    def predict_batch(parameters, batch):
        forward_logits = model_apply(
            jax, jnp, parameters, batch, None, 0.0, False, architecture
        )
        reverse_batch = batch[:, ::-1, ::-1]
        reverse_logits = model_apply(
            jax,
            jnp,
            parameters,
            reverse_batch,
            None,
            0.0,
            False,
            architecture,
        )
        return (
            jax.nn.sigmoid(forward_logits) + jax.nn.sigmoid(reverse_logits)
        ) / 2.0

    return predict_batch


def predict_probabilities(
    predict_batch,
    jnp,
    parameters,
    encoded_sequences: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for batch_indices, mask in padded_batches(indices, batch_size):
        probabilities = np.asarray(
            predict_batch(
                parameters, jnp.asarray(encoded_sequences[batch_indices])
            ),
            dtype=np.float64,
        )
        predictions.append(probabilities[mask.astype(bool)])
    return np.concatenate(predictions)


def flatten_parameters(parameters) -> dict[str, np.ndarray]:
    flattened: dict[str, np.ndarray] = {}
    for layer_name, layer in parameters.items():
        for parameter_name, value in layer.items():
            flattened[f"{layer_name}__{parameter_name}"] = np.asarray(value)
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
        for index, probability in zip(test_indices, probabilities, strict=True):
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


def train(args: argparse.Namespace) -> int:
    if args.sequence_length != 201:
        raise ValueError("This baseline is intentionally fixed to 201 bp.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("Dropout must be in [0, 1).")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("Learning rate must be finite and positive.")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("Weight decay must be finite and non-negative.")

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

    encoded_sequences = one_hot_encode(dataset["sequence"])
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
        f"Architecture: {args.architecture} "
        f"({architecture_parameter_count(args.architecture):,} parameters)"
    )

    key = jax.random.PRNGKey(args.seed)
    key, initialization_key = jax.random.split(key)
    with jax.default_device(device):
        parameters = initialize_parameters(
            jax, jnp, initialization_key, args.architecture
        )
        optimizer_state = initialize_adam(jax, jnp, parameters)

    @jax.jit
    def train_step(parameters, optimizer_state, batch_x, batch_y, batch_mask, key):
        augmentation_key, dropout_key = jax.random.split(key)
        reverse_mask = jax.random.bernoulli(
            augmentation_key, 0.5, shape=(batch_x.shape[0], 1, 1)
        )
        reverse_batch = batch_x[:, ::-1, ::-1]
        augmented = jnp.where(reverse_mask, reverse_batch, batch_x)

        def loss_function(candidate_parameters):
            logits = model_apply(
                jax,
                jnp,
                candidate_parameters,
                augmented,
                dropout_key,
                args.dropout,
                True,
                args.architecture,
            )
            losses = jnp.logaddexp(0.0, logits) - batch_y * logits
            loss = jnp.sum(losses * batch_mask) / jnp.sum(batch_mask)
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

    predict_batch = make_predict_batch(jax, jnp, args.architecture)
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
        for batch_number, (batch_indices, batch_mask) in enumerate(batches, start=1):
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
                    jnp.asarray(encoded_sequences[batch_indices]),
                    jnp.asarray(labels[batch_indices], dtype=jnp.float32),
                    jnp.asarray(batch_mask),
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
                    "Non-finite CNN state at "
                    f"epoch {epoch}, batch {batch_number}: loss={loss_value}, "
                    f"max_abs_logit={float(max_absolute_logit)}, "
                    f"gradient_norm={gradient_norm_value}, "
                    f"logits_finite={bool(logits_finite)}, "
                    f"gradients_finite={bool(gradients_finite)}, "
                    f"parameters_finite={bool(parameters_finite)}. "
                    f"First batch sample IDs: {failing_ids}. Set "
                    "JAX_DEBUG_NANS=True and JAX_DEBUG_INFS=True for an "
                    "operation-level traceback."
                )
            batch_losses.append(loss_value)
            batch_gradient_norms.append(gradient_norm_value)

        validation_probabilities = predict_probabilities(
            predict_batch,
            jnp,
            parameters,
            encoded_sequences,
            validation_indices,
            args.batch_size,
        )
        validation_metrics = classification_metrics(
            labels[validation_indices], validation_probabilities
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "training_batch_loss": float(np.mean(batch_losses)),
            "training_mean_gradient_norm": float(
                np.mean(batch_gradient_norms)
            ),
            "training_max_gradient_norm": float(
                np.max(batch_gradient_norms)
            ),
            **{
                f"internal_validation_{name}": value
                for name, value in validation_metrics.items()
            },
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d}: train_loss={np.mean(batch_losses):.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_auroc={validation_metrics['auroc']:.4f} "
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

    split_metrics: dict[str, dict[str, float]] = {}
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
            encoded_sequences,
            indices,
            args.batch_size,
        )
        split_probabilities[name] = probabilities
        split_metrics[name] = classification_metrics(labels[indices], probabilities)

    output_model = args.output_model.expanduser().resolve()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = output_model.with_suffix(output_model.suffix + ".tmp.npz")
    model_metadata = {
        "architecture": args.architecture,
        "specification": architecture_spec(args.architecture),
        "sequence_length": args.sequence_length,
        "dropout": args.dropout,
        "reverse_complement_augmentation": True,
        "reverse_complement_test_time_average": True,
        "best_epoch": best_epoch,
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

    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metrics = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    parameter_count = sum(
        int(np.prod(value.shape))
        for value in jax.tree_util.tree_leaves(best_parameters)
    )
    expected_parameter_count = architecture_parameter_count(args.architecture)
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Expected {expected_parameter_count:,} parameters, "
            f"but initialized {parameter_count:,}."
        )
    payload = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "model": str(output_model),
        "test_predictions": str(args.test_predictions.expanduser().resolve()),
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
    return 0


def main() -> int:
    return train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
