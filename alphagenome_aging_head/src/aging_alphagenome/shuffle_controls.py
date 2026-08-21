"""Generate mononucleotide- and exact-dinucleotide-preserving null datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from aging_alphagenome.composition import (
    adjacent_pair_counts,
    dinucleotide_counts,
    dinucleotide_shuffle,
    hamming_fraction,
    mononucleotide_shuffle,
    verify_dinucleotide_preservation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate null-control copies of a prepared aging-locus dataset. "
            "Mononucleotide shuffling preserves character counts; exact "
            "dinucleotide shuffling preserves every overlapping stride-1 pair."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-column", default="biological_sequence")
    parser.add_argument(
        "--mono-seed",
        type=int,
        default=17,
        help="Seed for the optional mononucleotide control. Use --no-mono to skip.",
    )
    parser.add_argument(
        "--no-mono",
        action="store_true",
        help="Do not generate the mononucleotide-shuffled control.",
    )
    parser.add_argument(
        "--dinucleotide-seeds",
        type=int,
        nargs="+",
        default=[17, 23, 41, 59, 73],
        help="Independent exact-dinucleotide shuffle seeds.",
    )
    return parser.parse_args()


def _open_text(path: Path):
    path = path.expanduser().resolve()
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_rows(
    path: Path, sequence_column: str
) -> tuple[list[dict[str, str]], list[str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Dataset has no header.")
        required = {"sample_id", sequence_column}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise ValueError("Dataset contains no rows.")
    return rows, fieldnames


def _write_rows(
    path: Path, rows: list[dict[str, str]], fieldnames: list[str]
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary(distances: list[float]) -> dict[str, float | int]:
    values = np.asarray(distances, dtype=np.float64)
    return {
        "rows": int(len(values)),
        "changed_rows": int(np.sum(values > 0.0)),
        "unchanged_rows": int(np.sum(values == 0.0)),
        "mean_hamming_fraction": float(values.mean()),
        "median_hamming_fraction": float(np.median(values)),
        "min_hamming_fraction": float(values.min()),
        "max_hamming_fraction": float(values.max()),
    }


def write_mononucleotide_control(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    sequence_column: str,
    seed: int,
) -> dict[str, float | int | str]:
    transformed: list[dict[str, str]] = []
    distances: list[float] = []
    for row in rows:
        original = row[sequence_column].upper()
        shuffled = mononucleotide_shuffle(
            original, seed=seed, key=row["sample_id"]
        )
        if sorted(original) != sorted(shuffled):
            raise RuntimeError(
                f"Mononucleotide shuffle changed character counts for {row['sample_id']}."
            )
        copied = dict(row)
        copied[sequence_column] = shuffled
        transformed.append(copied)
        distances.append(hamming_fraction(original, shuffled))
    _write_rows(path, transformed, fieldnames)
    return {
        "variant": "mononucleotide",
        "seed": seed,
        "path": str(path.expanduser().resolve()),
        **_summary(distances),
    }


def write_dinucleotide_control(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    sequence_column: str,
    seed: int,
) -> dict[str, float | int | str | bool]:
    transformed: list[dict[str, str]] = []
    distances: list[float] = []
    for row in rows:
        sample_id = row["sample_id"]
        original = row[sequence_column].upper()
        shuffled = dinucleotide_shuffle(original, seed=seed, key=sample_id)

        # This is deliberately redundant with dinucleotide_shuffle().  The
        # dataset writer is the experimental boundary, so it independently
        # verifies every row before anything is written to disk.
        verify_dinucleotide_preservation(original, shuffled)
        if dinucleotide_counts(original) != dinucleotide_counts(shuffled):
            raise RuntimeError(
                f"Canonical dinucleotide counts changed for {sample_id}."
            )
        if adjacent_pair_counts(original) != adjacent_pair_counts(shuffled):
            raise RuntimeError(f"Adjacent-pair counts changed for {sample_id}.")

        copied = dict(row)
        copied[sequence_column] = shuffled
        transformed.append(copied)
        distances.append(hamming_fraction(original, shuffled))

    _write_rows(path, transformed, fieldnames)
    return {
        "variant": "dinucleotide",
        "seed": seed,
        "path": str(path.expanduser().resolve()),
        "exact_pair_preservation_verified": True,
        **_summary(distances),
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_rows(dataset, args.sequence_column)
    outputs: list[dict[str, float | int | str | bool]] = []

    if not args.no_mono:
        outputs.append(
            write_mononucleotide_control(
                output_dir / f"aging_loci.mono_shuffled.seed_{args.mono_seed}.tsv",
                rows,
                fieldnames,
                args.sequence_column,
                args.mono_seed,
            )
        )

    seen: set[int] = set()
    for seed in args.dinucleotide_seeds:
        if seed in seen:
            raise ValueError(f"Duplicate dinucleotide seed: {seed}")
        seen.add(seed)
        outputs.append(
            write_dinucleotide_control(
                output_dir / f"aging_loci.dinuc_shuffled.seed_{seed}.tsv",
                rows,
                fieldnames,
                args.sequence_column,
                seed,
            )
        )

    manifest = {
        "source_dataset": str(dataset),
        "sequence_column": args.sequence_column,
        "definition": (
            "Dinucleotide shuffle preserves exact overlapping stride-1 adjacent-pair "
            "counts for every row, including all 16 canonical dinucleotides and CpG."
        ),
        "outputs": outputs,
    }
    manifest_path = output_dir / "shuffle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Source dataset: {dataset}")
    for output in outputs:
        print(
            f"{output['variant']:14s} seed={output['seed']}: "
            f"changed={output['changed_rows']}/{output['rows']} "
            f"mean_hamming={output['mean_hamming_fraction']:.3f} "
            f"-> {output['path']}"
        )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
