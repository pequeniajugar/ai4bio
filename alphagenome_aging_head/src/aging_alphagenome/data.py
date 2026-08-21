"""Prepare coordinate-only positive loci and random hg38 controls."""

from __future__ import annotations

import argparse
from bisect import bisect_left, insort
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
from typing import Protocol

from aging_alphagenome.composition import composition_match, gc_count


VALID_CHROMOSOME = re.compile(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y)$")


class ReferenceGenome(Protocol):
    """Minimal random-access reference interface used by the pipeline."""

    @property
    def lengths(self) -> Mapping[str, int]:
        ...

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        ...


class IndexedFasta:
    """pyfaidx-backed reference genome that does not load hg38 into RAM."""

    def __init__(self, path: Path):
        try:
            from pyfaidx import Fasta
        except ImportError as exc:
            raise RuntimeError(
                "pyfaidx is required. Activate the project environment."
            ) from exc

        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise ValueError(f"Reference FASTA does not exist: {self.path}")
        self._fasta = Fasta(
            str(self.path),
            as_raw=True,
            sequence_always_upper=True,
        )
        self._lengths = {
            str(name): len(self._fasta[name]) for name in self._fasta.keys()
        }

    @property
    def lengths(self) -> Mapping[str, int]:
        return self._lengths

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        if chromosome not in self._lengths:
            raise ValueError(f"{chromosome} is absent from the reference FASTA.")
        if start < 0 or end > self._lengths[chromosome] or start >= end:
            raise ValueError(
                f"Invalid interval {chromosome}:{start}-{end} for a contig of "
                f"length {self._lengths[chromosome]}."
            )
        return str(self._fasta[chromosome][start:end]).upper()


@dataclass
class Locus:
    """One positive or randomly sampled unlabeled locus."""

    sample_id: str
    pair_id: str
    chromosome: str
    position_1based: int
    start_0based: int
    end_0based: int
    reference: str
    alternate: str
    gene: str
    label: int
    label_type: str
    source: str
    split: str = ""
    gc_fraction_biological: float = 0.0
    biological_sequence: str = ""
    model_sequence: str = ""


def natural_chromosome_key(chromosome: str) -> tuple[int, str]:
    suffix = chromosome.removeprefix("chr")
    if suffix.isdigit():
        return int(suffix), ""
    return {"X": 23, "Y": 24}.get(suffix, 99), suffix


def centered_bounds(position_1based: int, length: int) -> tuple[int, int]:
    """Return a centered 0-based half-open interval.

    For odd lengths the locus is the exact middle base. For even lengths the
    locus is the first base of the right half, so its array index is length//2.
    """

    if position_1based < 1:
        raise ValueError("Positions must be 1-based positive integers.")
    if length < 1:
        raise ValueError("Window length must be positive.")
    center_0based = position_1based - 1
    start = center_0based - length // 2
    return start, start + length


def gc_fraction(sequence: str) -> float:
    sequence = sequence.upper()
    canonical = sum(sequence.count(base) for base in "ACGT")
    if canonical == 0:
        return 0.0
    return (sequence.count("G") + sequence.count("C")) / canonical


def n_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    return sum(base not in "ACGT" for base in sequence.upper()) / len(sequence)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_positive_loci(path: Path, *, split: str = "") -> list[Locus]:
    """Read only the first two TSV columns as chromosome and 1-based position."""

    path = path.expanduser().resolve()
    positives: list[Locus] = []
    coordinates: set[tuple[str, int]] = set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError("Input TSV has no header.")
        if len(header) < 2:
            raise ValueError("Input TSV must contain at least two columns.")

        id_prefix = f"POS_{split.upper()}" if split else "POS"
        for source_row, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"Line {source_row} has fewer than two columns.")
            chromosome = row[0].strip()
            if not VALID_CHROMOSOME.fullmatch(chromosome):
                raise ValueError(
                    f"Line {source_row} has unsupported chromosome "
                    f"{chromosome!r}."
                )
            try:
                position = int(row[1])
            except ValueError as exc:
                raise ValueError(
                    f"Line {source_row} has a non-integer position."
                ) from exc
            if position < 1:
                raise ValueError(f"Line {source_row} has position < 1.")

            coordinate = (chromosome, position)
            if coordinate in coordinates:
                raise ValueError(f"Duplicate positive coordinate: {coordinate}")
            coordinates.add(coordinate)

            positives.append(
                Locus(
                    sample_id=f"{id_prefix}_{source_row - 1:05d}",
                    pair_id=f"{id_prefix}_{source_row - 1:05d}",
                    chromosome=chromosome,
                    position_1based=position,
                    start_0based=position - 1,
                    end_0based=position,
                    reference="",
                    alternate="",
                    gene="",
                    label=1,
                    label_type="known_positive",
                    source=path.name,
                    split=split,
                )
            )

    if not positives:
        raise ValueError("Input TSV contains no loci.")
    return positives


def annotate_positive_sequences(
    positives: Sequence[Locus],
    reference: ReferenceGenome,
    biological_window: int,
    model_window: int,
) -> None:
    """Attach biological/model sequences using only coordinates and hg38."""

    for locus in positives:
        if locus.chromosome not in reference.lengths:
            raise ValueError(
                f"{locus.chromosome} is absent from the reference FASTA."
            )
        biological_start, biological_end = centered_bounds(
            locus.position_1based, biological_window
        )
        model_start, model_end = centered_bounds(
            locus.position_1based, model_window
        )
        biological_sequence = reference.fetch(
            locus.chromosome, biological_start, biological_end
        )
        model_sequence = reference.fetch(
            locus.chromosome, model_start, model_end
        )

        locus.reference = biological_sequence[biological_window // 2]
        locus.gc_fraction_biological = gc_fraction(biological_sequence)
        locus.biological_sequence = biological_sequence
        locus.model_sequence = model_sequence


def is_outside_radius(
    sorted_positions: Sequence[int], position: int, radius: int
) -> bool:
    index = bisect_left(sorted_positions, position)
    for neighbor_index in (index - 1, index):
        if 0 <= neighbor_index < len(sorted_positions):
            if abs(sorted_positions[neighbor_index] - position) <= radius:
                return False
    return True


def sample_matched_negatives(
    positives: Sequence[Locus],
    reference: ReferenceGenome,
    *,
    biological_window: int,
    model_window: int,
    negative_ratio: int,
    gc_tolerance: float,
    positive_exclusion_radius: int,
    negative_min_distance: int,
    max_attempts: int,
    seed: int,
    matching_strategy: str = "gc_tolerance",
    gc_count_tolerance: int = 0,
    base_fraction_tolerance: float = 0.02,
    dinucleotide_l1_tolerance: float = 0.15,
    max_n_fraction: float = 0.05,
) -> list[Locus]:
    """Sample controls matched by chromosome/REF and requested composition rule.

    ``random`` uses same-chromosome genomic background without composition
    matching. ``reference_only`` also matches the center reference base.
    ``gc_tolerance`` reproduces the original behavior. ``gc_exact`` matches
    the integer G+C count in the biological window (optionally within
    ``gc_count_tolerance`` bases). ``composition`` additionally constrains
    A/C/G/T fractions and the L1 distance between dinucleotide frequencies.
    """

    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1 when sampling.")
    if matching_strategy not in {
        "random",
        "reference_only",
        "gc_tolerance",
        "gc_exact",
        "composition",
    }:
        raise ValueError(
            "matching_strategy must be random, reference_only, gc_tolerance, "
            "gc_exact, or composition."
        )
    if gc_count_tolerance < 0:
        raise ValueError("gc_count_tolerance must be non-negative.")
    if not 0.0 <= base_fraction_tolerance <= 1.0:
        raise ValueError("base_fraction_tolerance must be between 0 and 1.")
    if not 0.0 <= dinucleotide_l1_tolerance <= 2.0:
        raise ValueError("dinucleotide_l1_tolerance must be between 0 and 2.")
    if not 0.0 <= max_n_fraction <= 1.0:
        raise ValueError("max_n_fraction must be between 0 and 1.")
    rng = random.Random(seed)
    positive_positions: dict[str, list[int]] = {}
    negative_positions: dict[str, list[int]] = {}
    for locus in positives:
        positive_positions.setdefault(locus.chromosome, []).append(
            locus.position_1based - 1
        )
    for positions in positive_positions.values():
        positions.sort()

    negatives: list[Locus] = []
    for positive in positives:
        chromosome = positive.chromosome
        contig_length = reference.lengths[chromosome]
        minimum_center = model_window // 2
        maximum_center = contig_length - (model_window - model_window // 2)
        if minimum_center > maximum_center:
            raise ValueError(
                f"{chromosome} is too short for a {model_window} bp window."
            )

        chromosome_negatives = negative_positions.setdefault(chromosome, [])
        for replicate in range(1, negative_ratio + 1):
            accepted: tuple[int, str, str] | None = None
            for _ in range(max_attempts):
                center_0based = rng.randint(minimum_center, maximum_center)
                if not is_outside_radius(
                    positive_positions[chromosome],
                    center_0based,
                    positive_exclusion_radius,
                ):
                    continue
                if not is_outside_radius(
                    chromosome_negatives,
                    center_0based,
                    negative_min_distance,
                ):
                    continue
                if matching_strategy != "random" and (
                    reference.fetch(
                        chromosome, center_0based, center_0based + 1
                    )
                    != positive.reference
                ):
                    continue

                biological_start = center_0based - biological_window // 2
                biological_end = biological_start + biological_window
                biological_sequence = reference.fetch(
                    chromosome, biological_start, biological_end
                )
                if n_fraction(biological_sequence) > max_n_fraction:
                    continue
                if matching_strategy in {"random", "reference_only"}:
                    pass
                elif matching_strategy == "gc_tolerance":
                    if (
                        abs(
                            gc_fraction(biological_sequence)
                            - positive.gc_fraction_biological
                        )
                        > gc_tolerance
                    ):
                        continue
                elif matching_strategy == "gc_exact":
                    if (
                        abs(
                            gc_count(biological_sequence)
                            - gc_count(positive.biological_sequence)
                        )
                        > gc_count_tolerance
                    ):
                        continue
                elif not composition_match(
                    biological_sequence,
                    positive.biological_sequence,
                    gc_count_tolerance=gc_count_tolerance,
                    base_fraction_tolerance=base_fraction_tolerance,
                    dinucleotide_l1_tolerance=dinucleotide_l1_tolerance,
                ):
                    continue

                model_start = center_0based - model_window // 2
                model_sequence = reference.fetch(
                    chromosome, model_start, model_start + model_window
                )
                if n_fraction(model_sequence) > max_n_fraction:
                    continue
                accepted = (
                    center_0based,
                    biological_sequence,
                    model_sequence,
                )
                break

            if accepted is None:
                raise RuntimeError(
                    "Could not sample a matched control for "
                    f"{positive.sample_id} after {max_attempts} attempts. "
                    "Increase --max-attempts or relax the selected matching tolerances."
                )

            center_0based, biological_sequence, model_sequence = accepted
            insort(chromosome_negatives, center_0based)
            negatives.append(
                Locus(
                    sample_id=(
                        f"NEG_{positive.sample_id.removeprefix('POS_')}_"
                        f"{replicate:02d}"
                    ),
                    pair_id=positive.sample_id,
                    chromosome=chromosome,
                    position_1based=center_0based + 1,
                    start_0based=center_0based,
                    end_0based=center_0based + 1,
                    reference=positive.reference,
                    alternate=positive.alternate,
                    gene="",
                    label=0,
                    label_type="random_genome_assumed_negative",
                    source=(
                        "hg38_random_matched"
                        if matching_strategy == "gc_tolerance"
                        else (
                            "hg38_random_unmatched"
                            if matching_strategy == "random"
                            else f"hg38_{matching_strategy}_matched"
                        )
                    ),
                    split=positive.split,
                    gc_fraction_biological=gc_fraction(biological_sequence),
                    biological_sequence=biological_sequence,
                    model_sequence=model_sequence,
                )
            )
    return negatives


def choose_validation_chromosomes(
    positive_counts: Mapping[str, int],
    validation_fraction: float,
    seed: int,
) -> tuple[str, ...]:
    """Select 2-4 whole chromosomes closest to the requested fraction."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    chromosomes = sorted(positive_counts, key=natural_chromosome_key)
    if len(chromosomes) < 2:
        raise ValueError("Chromosome holdout requires at least two chromosomes.")

    total = sum(positive_counts.values())
    target = total * validation_fraction
    minimum_size = 1 if len(chromosomes) < 4 else 2
    maximum_size = min(4, len(chromosomes) - 1)
    candidates: list[tuple[float, tuple[str, ...]]] = []
    best_distance = float("inf")

    for size in range(minimum_size, maximum_size + 1):
        for subset in itertools.combinations(chromosomes, size):
            subset_count = sum(positive_counts[chrom] for chrom in subset)
            distance = abs(subset_count - target)
            if distance < best_distance:
                best_distance = distance
                candidates = [(distance, subset)]
            elif distance == best_distance:
                candidates.append((distance, subset))

    def seeded_key(item: tuple[float, tuple[str, ...]]) -> str:
        _, subset = item
        value = f"{seed}|{','.join(subset)}".encode()
        return hashlib.sha256(value).hexdigest()

    return min(candidates, key=seeded_key)[1]


def assign_splits(
    loci: Sequence[Locus], validation_chromosomes: Sequence[str]
) -> None:
    validation_set = set(validation_chromosomes)
    for locus in loci:
        locus.split = (
            "validation"
            if locus.chromosome in validation_set
            else "train"
        )


def write_dataset(path: Path, loci: Sequence[Locus]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(asdict(loci[0]).keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for locus in loci:
            writer.writerow(asdict(locus))
    temporary.replace(path)


def write_manifest(path: Path, payload: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare positive aging loci and random controls from TSV "
            "coordinates and hg38."
        )
    )
    parser.add_argument("--input-tsv", type=Path)
    parser.add_argument(
        "--train-tsv",
        type=Path,
        help="Coordinate TSV whose rows receive split=train and label=1.",
    )
    parser.add_argument(
        "--validation-tsv",
        type=Path,
        help="Coordinate TSV whose rows receive split=validation and label=1.",
    )
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--biological-window", type=int, default=201)
    parser.add_argument("--model-window", type=int, default=2048)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument(
        "--matching-strategy",
        choices=(
            "random",
            "reference_only",
            "gc_tolerance",
            "gc_exact",
            "composition",
        ),
        default="gc_tolerance",
        help=(
            "Negative matching rule. random uses same-chromosome background; "
            "reference_only also matches the center base; gc_tolerance "
            "reproduces the original pipeline; gc_exact matches integer G+C "
            "count; composition also matches A/C/G/T fractions and "
            "dinucleotide frequencies."
        ),
    )
    parser.add_argument("--gc-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--gc-count-tolerance",
        type=int,
        default=0,
        help="Allowed difference in G+C base count for gc_exact/composition.",
    )
    parser.add_argument("--base-fraction-tolerance", type=float, default=0.02)
    parser.add_argument("--dinucleotide-l1-tolerance", type=float, default=0.15)
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--positive-exclusion-radius", type=int, default=2048)
    parser.add_argument("--negative-min-distance", type=int, default=201)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--validation-chromosomes",
        help="Optional comma-separated override, for example chr1,chr3.",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def validate_window_configuration(
    biological_window: int, model_window: int
) -> None:
    if biological_window < 1 or biological_window % 2 != 1:
        raise ValueError("--biological-window must be a positive odd number.")
    if (
        model_window < 2048
        or model_window > 2**20
        or model_window & (model_window - 1)
    ):
        raise ValueError(
            "--model-window must be a power of two from 2048 through 1048576."
        )
    if biological_window > model_window:
        raise ValueError("The biological window cannot exceed the model window.")


def main() -> int:
    args = parse_args()
    validate_window_configuration(args.biological_window, args.model_window)
    reference = IndexedFasta(args.reference_fasta)

    using_preassigned_split = (
        args.train_tsv is not None or args.validation_tsv is not None
    )
    if using_preassigned_split:
        if args.input_tsv is not None:
            raise ValueError(
                "Use either --input-tsv or --train-tsv/--validation-tsv, not both."
            )
        if args.train_tsv is None or args.validation_tsv is None:
            raise ValueError(
                "--train-tsv and --validation-tsv must be supplied together."
            )
        train_tsv = args.train_tsv.expanduser().resolve()
        validation_tsv = args.validation_tsv.expanduser().resolve()
        train_positives = read_positive_loci(train_tsv, split="train")
        validation_positives = read_positive_loci(
            validation_tsv, split="validation"
        )
        positives = train_positives + validation_positives
        duplicate_coordinates = (
            {
                (locus.chromosome, locus.position_1based)
                for locus in train_positives
            }
            & {
                (locus.chromosome, locus.position_1based)
                for locus in validation_positives
            }
        )
        if duplicate_coordinates:
            examples = sorted(duplicate_coordinates)[:10]
            raise ValueError(
                "Train and validation TSVs overlap at coordinates: "
                f"{examples}"
            )
        input_manifest: object = {
            "train": {
                "path": str(train_tsv),
                "sha256": sha256_file(train_tsv),
                "positive_rows": len(train_positives),
            },
            "validation": {
                "path": str(validation_tsv),
                "sha256": sha256_file(validation_tsv),
                "positive_rows": len(validation_positives),
            },
        }
    else:
        if args.input_tsv is None:
            raise ValueError(
                "Provide --input-tsv or both --train-tsv and --validation-tsv."
            )
        input_tsv = args.input_tsv.expanduser().resolve()
        positives = read_positive_loci(input_tsv)
        input_manifest = {
            "path": str(input_tsv),
            "sha256": sha256_file(input_tsv),
            "positive_rows": len(positives),
        }

    annotate_positive_sequences(
        positives,
        reference,
        args.biological_window,
        args.model_window,
    )
    negatives = (
        sample_matched_negatives(
            positives,
            reference,
            biological_window=args.biological_window,
            model_window=args.model_window,
            negative_ratio=args.negative_ratio,
            gc_tolerance=args.gc_tolerance,
            positive_exclusion_radius=args.positive_exclusion_radius,
            negative_min_distance=args.negative_min_distance,
            max_attempts=args.max_attempts,
            seed=args.seed,
            matching_strategy=args.matching_strategy,
            gc_count_tolerance=args.gc_count_tolerance,
            base_fraction_tolerance=args.base_fraction_tolerance,
            dinucleotide_l1_tolerance=args.dinucleotide_l1_tolerance,
            max_n_fraction=args.max_n_fraction,
        )
        if args.negative_ratio
        else []
    )

    positive_counts = Counter(locus.chromosome for locus in positives)
    if using_preassigned_split:
        validation_chromosomes = tuple(
            sorted(
                {locus.chromosome for locus in validation_positives},
                key=natural_chromosome_key,
            )
        )
        split_strategy = "preassigned_coordinate_files"
    elif args.validation_chromosomes:
        validation_chromosomes = tuple(
            value.strip()
            for value in args.validation_chromosomes.split(",")
            if value.strip()
        )
        unknown = set(validation_chromosomes) - set(positive_counts)
        if unknown:
            raise ValueError(
                f"Validation chromosomes have no positives: {sorted(unknown)}"
            )
        split_strategy = "whole_chromosome_holdout"
    else:
        validation_chromosomes = choose_validation_chromosomes(
            positive_counts,
            args.validation_fraction,
            args.seed,
        )
        split_strategy = "whole_chromosome_holdout"

    loci = positives + negatives
    if not using_preassigned_split:
        assign_splits(loci, validation_chromosomes)
    random.Random(args.seed).shuffle(loci)
    write_dataset(args.output_tsv, loci)

    split_label_counts = Counter((locus.split, locus.label) for locus in loci)
    manifest_path = args.manifest or args.output_tsv.with_suffix(
        args.output_tsv.suffix + ".manifest.json"
    )
    manifest = {
        "input": input_manifest,
        "reference": {
            "path": str(args.reference_fasta.expanduser().resolve()),
            "assembly": "GRCh38/hg38",
            "used_contig_lengths": {
                chrom: reference.lengths[chrom]
                for chrom in sorted(positive_counts, key=natural_chromosome_key)
            },
        },
        "parameters": {
            "biological_window": args.biological_window,
            "model_window": args.model_window,
            "negative_ratio": args.negative_ratio,
            "matching_strategy": args.matching_strategy,
            "gc_tolerance": args.gc_tolerance,
            "gc_count_tolerance": args.gc_count_tolerance,
            "base_fraction_tolerance": args.base_fraction_tolerance,
            "dinucleotide_l1_tolerance": args.dinucleotide_l1_tolerance,
            "max_n_fraction": args.max_n_fraction,
            "positive_exclusion_radius": args.positive_exclusion_radius,
            "negative_min_distance": args.negative_min_distance,
            "validation_fraction_requested": args.validation_fraction,
            "seed": args.seed,
        },
        "split": {
            "strategy": split_strategy,
            "validation_chromosomes": list(validation_chromosomes),
            "counts": {
                f"{split}_label_{label}": count
                for (split, label), count in sorted(split_label_counts.items())
            },
        },
        "output": {
            "path": str(args.output_tsv.expanduser().resolve()),
            "rows": len(loci),
        },
        "label_semantics": {
            "1": "known aging-associated positive",
            "0": (
                "random hg38 context used as an assumed negative for the "
                "baseline classifier"
            ),
        },
        "column_policy": (
            "Only TSV columns 1 and 2 were read as chromosome and 1-based "
            "position; all remaining source columns were ignored."
        ),
    }
    write_manifest(manifest_path, manifest)

    validation_positive_count = sum(
        locus.split == "validation" and locus.label == 1 for locus in loci
    )
    print(f"Prepared {len(positives):,} positives and {len(negatives):,} controls.")
    print(
        "Validation chromosomes: "
        f"{','.join(validation_chromosomes)} "
        f"({validation_positive_count / len(positives):.1%} of positives)"
    )
    print(f"Dataset:  {args.output_tsv.expanduser().resolve()}")
    print(f"Manifest: {manifest_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
