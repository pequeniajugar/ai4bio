#!/usr/bin/env python3
"""Validate the input data and an AlphaGenome installation on an HPC node."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TSV = PROJECT_ROOT / "RS_PDL50.wgs.rediportal.vcf.isec.tsv"
VALID_CHROMOSOME = re.compile(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M|MT)$")
VALID_ALLELE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate only the TSV; do not import JAX or AlphaGenome.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless JAX exposes at least one CUDA GPU.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local Orbax checkpoint directory to restore.",
    )
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Run a 2,048 bp prediction after restoring --checkpoint.",
    )
    return parser.parse_args()


def validate_tsv(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"TSV does not exist: {path}")

    required_columns = {"#CHROM", "POS"}
    coordinates: set[tuple[str, int]] = set()
    chromosomes: Counter[str] = Counter()
    allele_changes: Counter[str] = Counter()
    gene_tokens: set[str] = set()
    row_count = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSV has no header.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"TSV is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if None in row:
                raise ValueError(f"Line {line_number} has too many fields.")

            chromosome = (row.get("#CHROM") or "").strip()
            if not VALID_CHROMOSOME.fullmatch(chromosome):
                raise ValueError(
                    f"Line {line_number} has unsupported chromosome "
                    f"{chromosome!r}."
                )

            try:
                position = int((row.get("POS") or "").strip())
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number} has a non-integer POS."
                ) from exc
            if position < 1:
                raise ValueError(f"Line {line_number} has POS < 1.")

            coordinate = (chromosome, position)
            if coordinate in coordinates:
                raise ValueError(
                    f"Duplicate coordinate at line {line_number}: {coordinate}"
                )
            coordinates.add(coordinate)
            chromosomes[chromosome] += 1

            reference = (row.get("REF") or "").strip().upper()
            alternate = (row.get("ALT") or "").strip().upper()
            if reference or alternate:
                if not (
                    VALID_ALLELE.fullmatch(reference)
                    and VALID_ALLELE.fullmatch(alternate)
                ):
                    raise ValueError(
                        f"Line {line_number} has invalid REF/ALT alleles."
                    )
                allele_changes[f"{reference}>{alternate}"] += 1

            for gene in (row.get("Gene.refGene") or "").split(";"):
                gene = gene.strip()
                if gene and gene != "-":
                    gene_tokens.add(gene)

    if row_count == 0:
        raise ValueError("TSV contains no data rows.")

    return {
        "path": str(path),
        "rows": row_count,
        "columns": len(reader.fieldnames),
        "unique_coordinates": len(coordinates),
        "chromosomes": dict(sorted(chromosomes.items())),
        "allele_changes": dict(sorted(allele_changes.items())),
        "unique_gene_tokens": len(gene_tokens),
        "coordinate_system": "input POS is 1-based; SNV interval is [POS-1, POS)",
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def print_nvidia_smi() -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"nvidia-smi unavailable: {exc}")
        return

    if result.returncode == 0:
        print("nvidia-smi GPU(s):")
        print(result.stdout.strip())
    else:
        print(f"nvidia-smi failed: {result.stderr.strip()}")


def import_runtime():
    # TensorFlow is used by the upstream data loader only. Prevent it from
    # reserving GPU memory that belongs to JAX.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except RuntimeError as exc:
        raise RuntimeError(
            "TensorFlow initialized an accelerator before it could be hidden."
        ) from exc

    import jax
    import jax.numpy as jnp
    import alphagenome
    import alphagenome_research
    from alphagenome_research.model import dna_model

    del alphagenome, alphagenome_research
    return jax, jnp, dna_model


def run_jax_check(jax, jnp, require_gpu: bool):
    print_nvidia_smi()
    devices = jax.devices()
    print("JAX devices:")
    for device in devices:
        print(f"  {device}")

    gpu_devices = [device for device in devices if device.platform == "gpu"]
    if require_gpu and not gpu_devices:
        raise RuntimeError("JAX does not expose a CUDA GPU.")

    selected_device = gpu_devices[0] if gpu_devices else devices[0]
    values = jax.device_put(jnp.arange(4096, dtype=jnp.float32), selected_device)
    checksum = jnp.sum(values * values).block_until_ready()
    expected = sum(float(value * value) for value in range(4096))
    if abs(float(checksum) - expected) / expected > 1e-5:
        raise RuntimeError("The JAX device computation returned a bad result.")
    print(
        f"JAX computation passed on {selected_device}; "
        f"checksum={float(checksum):.1f}"
    )
    return selected_device


def offline_organism_settings(dna_model):
    settings = {}
    for organism, organism_settings in dna_model.default_organism_settings().items():
        settings[organism] = dataclasses.replace(
            organism_settings,
            fasta_path=None,
            gtf_feather_path=None,
            pas_feather_path=None,
            splice_site_starts_feather_path=None,
            splice_site_ends_feather_path=None,
            calibration_path=None,
        )
    return settings


def load_model_and_optionally_predict(
    dna_model,
    checkpoint: Path,
    selected_device,
    run_inference: bool,
) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    required_files = [
        checkpoint / "_CHECKPOINT_METADATA",
        checkpoint / "manifest.ocdbt",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(
            "Checkpoint is missing required Orbax files: " + ", ".join(missing)
        )

    print(f"Restoring AlphaGenome checkpoint from {checkpoint}")
    started = time.monotonic()
    model = dna_model.create(
        checkpoint,
        organism_settings=offline_organism_settings(dna_model),
        device=selected_device,
    )
    print(f"Checkpoint restored in {time.monotonic() - started:.1f} seconds.")

    if not run_inference:
        return

    sequence = "ACGT" * 512
    print(f"Running AlphaGenome inference on {len(sequence):,} bp.")
    started = time.monotonic()
    output = model.predict_sequence(
        sequence,
        requested_outputs=[dna_model.OutputType.SPLICE_SITES],
        ontology_terms=None,
    )
    if output.splice_sites is None:
        raise RuntimeError("Inference returned no splice-site output.")
    shape = tuple(output.splice_sites.values.shape)
    if shape[0] != len(sequence):
        raise RuntimeError(
            f"Unexpected splice-site output shape {shape} for {len(sequence)} bp."
        )
    print(
        f"Inference passed in {time.monotonic() - started:.1f} seconds; "
        f"splice-site output shape={shape}."
    )


def main() -> int:
    args = parse_args()
    if args.run_inference and args.checkpoint is None:
        print("--run-inference requires --checkpoint.", file=sys.stderr)
        return 2
    if args.data_only and (
        args.require_gpu or args.checkpoint is not None or args.run_inference
    ):
        print(
            "--data-only cannot be combined with GPU or checkpoint options.",
            file=sys.stderr,
        )
        return 2

    try:
        summary = validate_tsv(args.tsv)
        print("Dataset validation passed:")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.data_only:
            print("DATA-ONLY SMOKE TEST PASSED")
            return 0

        jax, jnp, dna_model = import_runtime()
        print(
            "Runtime versions: "
            f"alphagenome={package_version('alphagenome')}, "
            f"alphagenome_research={package_version('alphagenome_research')}, "
            f"jax={package_version('jax')}, "
            f"jaxlib={package_version('jaxlib')}, "
            f"tensorflow={package_version('tensorflow')}"
        )
        selected_device = run_jax_check(jax, jnp, args.require_gpu)

        if args.checkpoint is not None:
            load_model_and_optionally_predict(
                dna_model,
                args.checkpoint,
                selected_device,
                args.run_inference,
            )
        else:
            print("Checkpoint restore skipped (no --checkpoint supplied).")

        print("ALPHAGENOME HPC SMOKE TEST PASSED")
        return 0
    except Exception as exc:  # A smoke test should report a compact job-log error.
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
