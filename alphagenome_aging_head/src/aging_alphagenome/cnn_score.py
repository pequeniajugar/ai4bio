"""Score a saved CNN checkpoint on the chromosome-held-out split of a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aging_alphagenome.cnn import (
    load_jax,
    make_predict_batch,
    predict_probabilities,
    read_dataset,
    write_test_predictions,
)
from aging_alphagenome.features import one_hot_encode
from aging_alphagenome.head import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved 201-bp CNN checkpoint on the validation/test split "
            "of a prepared dataset without retraining."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless JAX exposes a GPU.",
    )
    return parser.parse_args()


def load_model(path: Path, jnp):
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"CNN checkpoint lacks metadata_json: {path}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        parameters: dict[str, dict[str, object]] = {}
        for name in archive.files:
            if name == "metadata_json":
                continue
            if "__" not in name:
                raise ValueError(f"Unexpected parameter key in checkpoint: {name}")
            layer_name, parameter_name = name.split("__", 1)
            parameters.setdefault(layer_name, {})[parameter_name] = jnp.asarray(
                archive[name]
            )
    return parameters, metadata


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    jax, jnp = load_jax()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if args.require_gpu and not gpu_devices:
        raise RuntimeError("JAX exposes no GPU in this job.")
    device = gpu_devices[0] if gpu_devices else devices[0]

    with jax.default_device(device):
        parameters, metadata = load_model(args.model, jnp)

    sequence_length = int(metadata.get("sequence_length", 201))
    architecture = str(metadata["architecture"])
    dataset = read_dataset(args.dataset, sequence_length)
    test_indices = np.flatnonzero(dataset["split"] == "validation")
    labels = dataset["label"]
    if set(np.unique(labels[test_indices])) != {0, 1}:
        raise ValueError("Test split must contain both labels.")

    encoded_sequences = one_hot_encode(dataset["sequence"])
    predict_batch = make_predict_batch(jax, jnp, architecture)
    probabilities = predict_probabilities(
        predict_batch,
        jnp,
        parameters,
        encoded_sequences,
        test_indices,
        args.batch_size,
    )
    metrics = classification_metrics(labels[test_indices], probabilities)

    write_test_predictions(
        args.test_predictions,
        dataset,
        test_indices,
        probabilities,
    )

    metrics_path = args.metrics.expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "model": str(args.model.expanduser().resolve()),
        "device": str(device),
        "count": int(len(test_indices)),
        "architecture": metadata,
        "metrics": metrics,
        "purpose": (
            "Inference-only perturbation test: the checkpoint is fixed and only "
            "the evaluated sequence representation changes."
        ),
    }
    temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metrics_path)

    print(f"Using {device}")
    print(json.dumps({"test": metrics}, indent=2))
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {args.test_predictions.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
