"""Extract compact frozen AlphaGenome trunk features on a GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("combined", "center-1bp", "pooled-128bp"),
        default="combined",
    )
    parser.add_argument("--biological-window", type=int, default=201)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU extraction for debugging; full extraction is impractical.",
    )
    return parser.parse_args()


def read_dataset(path: Path, limit: int | None) -> dict[str, np.ndarray]:
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
            "label_type",
            "split",
            "biological_sequence",
            "model_sequence",
        }
        if reader.fieldnames is None:
            raise ValueError("Prepared dataset has no header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Prepared dataset is missing columns: {sorted(missing)}"
            )
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError("Prepared dataset contains no rows.")

    sequence_lengths = {len(row["model_sequence"]) for row in rows}
    if len(sequence_lengths) != 1:
        raise ValueError("model_sequence lengths are inconsistent.")
    sequence_length = sequence_lengths.pop()
    biological_lengths = {len(row["biological_sequence"]) for row in rows}
    if len(biological_lengths) != 1:
        raise ValueError("biological_sequence lengths are inconsistent.")
    biological_length = biological_lengths.pop()
    if sequence_length < 2048 or sequence_length & (sequence_length - 1):
        raise ValueError(
            "model_sequence length must be a power of two and at least 2048."
        )
    for row in rows:
        if set(row["model_sequence"].upper()) - set("ACGTN"):
            raise ValueError(f"{row['sample_id']} has an invalid DNA sequence.")

    def strings(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=str)

    return {
        "sample_id": strings("sample_id"),
        "pair_id": strings("pair_id"),
        "chromosome": strings("chromosome"),
        "position_1based": np.asarray(
            [int(row["position_1based"]) for row in rows], dtype=np.int64
        ),
        "label": np.asarray([int(row["label"]) for row in rows], dtype=np.int8),
        "label_type": strings("label_type"),
        "split": strings("split"),
        "biological_sequence": strings("biological_sequence"),
        "sequence": strings("model_sequence"),
        "biological_length": np.asarray(biological_length, dtype=np.int64),
        "sequence_length": np.asarray(sequence_length, dtype=np.int64),
    }


def one_hot_encode(sequences: np.ndarray) -> np.ndarray:
    """Encode A/C/G/T as four channels and N as all zeros."""

    batch_size = len(sequences)
    sequence_length = len(sequences[0])
    encoded = np.zeros((batch_size, sequence_length, 4), dtype=np.float32)
    for row_index, sequence in enumerate(sequences):
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        for column, base in enumerate(b"ACGT"):
            encoded[row_index, :, column] = sequence_bytes == base
    return encoded


def pooling_weights(
    model_length: int, biological_window: int, bin_size: int = 128
) -> np.ndarray:
    """Return overlap weights for bins covering the central biological window."""

    if biological_window % 2 != 1:
        raise ValueError("biological_window must be odd.")
    center = model_length // 2
    biological_start = center - biological_window // 2
    biological_end = biological_start + biological_window
    num_bins = model_length // bin_size
    weights = np.zeros(num_bins, dtype=np.float32)
    for index in range(num_bins):
        bin_start = index * bin_size
        bin_end = bin_start + bin_size
        weights[index] = max(
            0, min(bin_end, biological_end) - max(bin_start, biological_start)
        )
    if weights.sum() != biological_window:
        raise ValueError("Could not align the biological and embedding windows.")
    return weights / weights.sum()


def validate_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    required = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "manifest.ocdbt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Checkpoint is missing required Orbax files: " + ", ".join(missing)
        )
    return checkpoint


def load_runtime():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.set_visible_devices([], "TPU")

    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome_research.model import dna_model
    from alphagenome_research.model.metadata import metadata as metadata_lib

    return jax, jnp, ocp, dna_model, metadata_lib


def extract_features(args: argparse.Namespace) -> None:
    dataset = read_dataset(args.dataset, args.limit)
    checkpoint = validate_checkpoint(args.checkpoint)
    biological_length = int(dataset.pop("biological_length"))
    if biological_length != args.biological_window:
        raise ValueError(
            "Dataset biological_sequence length is "
            f"{biological_length}, but --biological-window is "
            f"{args.biological_window}."
        )
    sequence_length = int(dataset["sequence_length"])
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    jax, jnp, ocp, dna_model, metadata_lib = load_runtime()
    devices = jax.devices()
    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if not gpu_devices and not args.allow_cpu:
        raise RuntimeError(
            "JAX exposes no GPU. Use a GPU job or --allow-cpu only for debugging."
        )
    device = gpu_devices[0] if gpu_devices else devices[0]
    print(f"Using {device}")

    organisms = (
        dna_model.Organism.HOMO_SAPIENS,
        dna_model.Organism.MUS_MUSCULUS,
    )
    metadata = {organism: metadata_lib.load(organism) for organism in organisms}
    init_fn, _, trunk_apply_fn, _, _ = dna_model.create_model(metadata)
    shape_sequence = jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32)
    shape_organism = jax.ShapeDtypeStruct((1,), jnp.int32)
    target_shapes = jax.eval_shape(
        init_fn,
        jax.random.PRNGKey(0),
        shape_sequence,
        shape_organism,
    )

    print(f"Restoring backbone from {checkpoint}")
    with jax.default_device(device):
        params, state = ocp.StandardCheckpointer().restore(
            checkpoint,
            target=target_shapes,
            strict=True,
        )

    weights_128bp = jnp.asarray(
        pooling_weights(sequence_length, args.biological_window)
    )
    center_index = sequence_length // 2
    feature_mode = args.feature_mode

    def apply_and_pool(params, state, dna_sequence, organism_index):
        embeddings = trunk_apply_fn(
            params, state, dna_sequence, organism_index
        )
        pieces = []
        if feature_mode in ("combined", "center-1bp"):
            pieces.append(embeddings.embeddings_1bp[:, center_index, :])
        if feature_mode in ("combined", "pooled-128bp"):
            pooled = jnp.einsum(
                "bsd,s->bd",
                embeddings.embeddings_128bp,
                weights_128bp,
            )
            pieces.append(pooled)
        return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=-1)

    apply_and_pool = jax.jit(apply_and_pool)
    sequence_values = dataset.pop("sequence")
    num_rows = len(sequence_values)
    features: list[np.ndarray] = []
    started = time.monotonic()

    for start in range(0, num_rows, args.batch_size):
        stop = min(start + args.batch_size, num_rows)
        sequence_batch = sequence_values[start:stop]
        actual_batch_size = len(sequence_batch)
        if actual_batch_size < args.batch_size:
            padding = np.repeat(sequence_batch[-1:], args.batch_size - stop + start)
            sequence_batch = np.concatenate([sequence_batch, padding])

        encoded = one_hot_encode(sequence_batch)
        organism_index = np.zeros(args.batch_size, dtype=np.int32)
        with jax.default_device(device):
            batch_features = apply_and_pool(
                params,
                state,
                jax.device_put(encoded, device),
                jax.device_put(organism_index, device),
            )
            batch_features = np.asarray(
                batch_features[:actual_batch_size], dtype=np.float32
            )
        features.append(batch_features)

        if stop == num_rows or stop % 100 == 0:
            elapsed = time.monotonic() - started
            print(
                f"Extracted {stop:,}/{num_rows:,} examples "
                f"({stop / max(elapsed, 1e-6):.2f} examples/s)"
            )

    feature_matrix = np.concatenate(features, axis=0)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp.npz")
    np.savez_compressed(temporary, features=feature_matrix, **dataset)
    temporary.replace(output)

    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.expanduser().resolve()),
                "checkpoint": str(checkpoint),
                "feature_mode": args.feature_mode,
                "biological_window": args.biological_window,
                "model_window": sequence_length,
                "examples": num_rows,
                "feature_dimension": feature_matrix.shape[1],
                "device": str(device),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Features saved to {output} with shape {feature_matrix.shape}")


def main() -> int:
    args = parse_args()
    extract_features(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
