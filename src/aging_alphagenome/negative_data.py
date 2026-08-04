"""Build a large coordinate-only corpus of random non-aging regions."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import random
from typing import TextIO

from aging_alphagenome.data import (
    IndexedFasta,
    Locus,
    ReferenceGenome,
    VALID_CHROMOSOME,
    centered_bounds,
    n_fraction,
    natural_chromosome_key,
    read_positive_loci,
    sha256_file,
)


@dataclass(frozen=True, order=True)
class GenomicInterval:
    """A 0-based, half-open genomic interval."""

    chromosome: str
    start: int
    end: int


@dataclass(frozen=True)
class EligibleStartRun:
    """A half-open run of starts that produce valid model intervals."""

    chromosome: str
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass
class SamplingStats:
    attempted: int = 0
    rejected_ambiguous_sequence: int = 0


@dataclass(frozen=True)
class NegativeRegion:
    chromosome: str
    start: int
    end: int
    sequence: str = ""


def merge_intervals(
    intervals: Sequence[GenomicInterval],
) -> list[GenomicInterval]:
    """Merge overlapping or directly adjacent intervals by chromosome."""

    merged: list[GenomicInterval] = []
    for interval in sorted(
        intervals,
        key=lambda value: (
            natural_chromosome_key(value.chromosome),
            value.chromosome,
            value.start,
            value.end,
        ),
    ):
        if interval.start < 0 or interval.end <= interval.start:
            raise ValueError(f"Invalid interval: {interval}")
        if (
            merged
            and merged[-1].chromosome == interval.chromosome
            and interval.start <= merged[-1].end
        ):
            previous = merged[-1]
            merged[-1] = GenomicInterval(
                previous.chromosome,
                previous.start,
                max(previous.end, interval.end),
            )
        else:
            merged.append(interval)
    return merged


def build_aging_mask(
    positives: Sequence[Locus],
    contig_lengths: Mapping[str, int],
    mask_window: int,
) -> list[GenomicInterval]:
    """Return merged, clipped contexts centered on every known positive."""

    if mask_window < 1:
        raise ValueError("mask_window must be positive.")

    intervals: list[GenomicInterval] = []
    for positive in positives:
        if positive.chromosome not in contig_lengths:
            raise ValueError(
                f"{positive.chromosome} is absent from the reference FASTA."
            )
        contig_length = contig_lengths[positive.chromosome]
        if positive.position_1based > contig_length:
            raise ValueError(
                f"{positive.chromosome}:{positive.position_1based} is outside "
                f"its {contig_length:,} bp reference contig."
            )
        start, end = centered_bounds(positive.position_1based, mask_window)
        intervals.append(
            GenomicInterval(
                positive.chromosome,
                max(start, 0),
                min(end, contig_length),
            )
        )
    return merge_intervals(intervals)


def build_eligible_start_runs(
    contig_lengths: Mapping[str, int],
    masks: Sequence[GenomicInterval],
    region_length: int,
    chromosomes: Sequence[str],
) -> list[EligibleStartRun]:
    """Find all starts whose full region does not overlap an aging mask.

    A mask ``[m_start, m_end)`` invalidates every integer region start in
    ``[m_start - region_length + 1, m_end)``. Taking the complement of those
    invalid starts avoids expanding a multi-gigabase, base-level mask.
    """

    if region_length < 1:
        raise ValueError("region_length must be positive.")

    masks_by_chromosome: dict[str, list[GenomicInterval]] = {}
    for mask in masks:
        masks_by_chromosome.setdefault(mask.chromosome, []).append(mask)

    runs: list[EligibleStartRun] = []
    for chromosome in chromosomes:
        if chromosome not in contig_lengths:
            raise ValueError(
                f"Requested chromosome {chromosome} is absent from the FASTA."
            )
        contig_length = contig_lengths[chromosome]
        number_of_starts = contig_length - region_length + 1
        if number_of_starts <= 0:
            continue

        invalid_starts = merge_intervals(
            [
                GenomicInterval(
                    chromosome,
                    max(0, mask.start - region_length + 1),
                    min(number_of_starts, mask.end),
                )
                for mask in masks_by_chromosome.get(chromosome, [])
                if max(0, mask.start - region_length + 1)
                < min(number_of_starts, mask.end)
            ]
        )
        cursor = 0
        for invalid in invalid_starts:
            if cursor < invalid.start:
                runs.append(
                    EligibleStartRun(chromosome, cursor, invalid.start)
                )
            cursor = max(cursor, invalid.end)
        if cursor < number_of_starts:
            runs.append(
                EligibleStartRun(chromosome, cursor, number_of_starts)
            )
    return runs


def _region_for_rank(
    runs: Sequence[EligibleStartRun],
    cumulative_counts: Sequence[int],
    rank: int,
    region_length: int,
) -> NegativeRegion:
    run_index = bisect_right(cumulative_counts, rank)
    preceding = 0 if run_index == 0 else cumulative_counts[run_index - 1]
    run = runs[run_index]
    start = run.start + rank - preceding
    return NegativeRegion(run.chromosome, start, start + region_length)


def sample_negative_regions(
    runs: Sequence[EligibleStartRun],
    *,
    number: int,
    region_length: int,
    reference: ReferenceGenome,
    max_n_fraction: float,
    max_attempts_per_sample: int,
    seed: int,
    include_sequence: bool = False,
    stats: SamplingStats | None = None,
) -> Iterator[NegativeRegion]:
    """Uniformly sample unique eligible starts, optionally retaining sequence."""

    if number < 1:
        raise ValueError("number must be positive.")
    if not 0.0 <= max_n_fraction <= 1.0:
        raise ValueError("max_n_fraction must be between 0 and 1.")
    if max_attempts_per_sample < 1:
        raise ValueError("max_attempts_per_sample must be positive.")

    cumulative_counts: list[int] = []
    total = 0
    for run in runs:
        total += run.count
        cumulative_counts.append(total)
    if number > total:
        raise ValueError(
            f"Requested {number:,} rows but only {total:,} eligible starts exist."
        )

    sampling_stats = stats if stats is not None else SamplingStats()
    rng = random.Random(seed)
    # A lazy partial Fisher-Yates shuffle draws ranks without replacement
    # without materializing the multi-billion-element sampling space.
    rank_swaps: dict[int, int] = {}
    remaining_ranks = total
    accepted = 0
    maximum_attempts = min(total, number * max_attempts_per_sample)

    while accepted < number and sampling_stats.attempted < maximum_attempts:
        chosen_index = rng.randrange(remaining_ranks)
        rank = rank_swaps.get(chosen_index, chosen_index)
        remaining_ranks -= 1
        rank_swaps[chosen_index] = rank_swaps.get(
            remaining_ranks, remaining_ranks
        )
        sampling_stats.attempted += 1

        region = _region_for_rank(
            runs, cumulative_counts, rank, region_length
        )
        sequence = reference.fetch(
            region.chromosome, region.start, region.end
        )
        if n_fraction(sequence) > max_n_fraction:
            sampling_stats.rejected_ambiguous_sequence += 1
            continue

        accepted += 1
        yield NegativeRegion(
            region.chromosome,
            region.start,
            region.end,
            sequence if include_sequence else "",
        )

    if accepted < number:
        raise RuntimeError(
            f"Accepted only {accepted:,}/{number:,} regions after "
            f"{sampling_stats.attempted:,} unique attempts. Increase "
            "--max-attempts-per-sample or --max-n-fraction."
        )


def _open_output(path: Path, temporary: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(temporary, "wt", encoding="utf-8", newline="")
    return temporary.open("w", encoding="utf-8", newline="")


def write_mask_bed(path: Path, masks: Sequence[GenomicInterval]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for index, mask in enumerate(masks, start=1):
            writer.writerow(
                [
                    mask.chromosome,
                    mask.start,
                    mask.end,
                    f"AGING_MASK_{index:06d}",
                ]
            )
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


def parse_chromosomes(
    value: str | None, contig_lengths: Mapping[str, int]
) -> tuple[str, ...]:
    if value:
        chromosomes = tuple(
            chromosome.strip()
            for chromosome in value.split(",")
            if chromosome.strip()
        )
        if len(set(chromosomes)) != len(chromosomes):
            raise ValueError("--chromosomes contains duplicate values.")
    else:
        chromosomes = tuple(
            sorted(
                (
                    chromosome
                    for chromosome in contig_lengths
                    if VALID_CHROMOSOME.fullmatch(chromosome)
                ),
                key=natural_chromosome_key,
            )
        )
    if not chromosomes:
        raise ValueError("No chromosomes were selected.")
    missing = set(chromosomes) - set(contig_lengths)
    if missing:
        raise ValueError(
            f"Chromosomes are absent from the FASTA: {sorted(missing)}"
        )
    return chromosomes


def default_mask_path(output: Path, mask_window: int) -> Path:
    name = output.name
    if name.endswith(".tsv.gz"):
        stem = name[:-7]
    else:
        stem = output.stem
    return output.with_name(f"{stem}.aging_mask_{mask_window}bp.bed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mask known aging contexts and uniformly sample coordinate-only "
            "AlphaGenome-sized negative regions from the remaining hg38 space."
        )
    )
    parser.add_argument("--train-tsv", type=Path, required=True)
    parser.add_argument("--validation-tsv", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--mask-bed", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--number", type=int, default=1_000_000)
    parser.add_argument("--mask-window", type=int, default=200)
    parser.add_argument("--region-length", type=int, default=16_384)
    parser.add_argument(
        "--chromosomes",
        help=(
            "Comma-separated reference contigs. By default, use "
            "chr1-chr22,chrX,chrY when present."
        ),
    )
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    parser.add_argument("--max-attempts-per-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        help=(
            "Add model_sequence to each row. This is off by default because "
            "one million 16,384 bp strings require at least 16.4 GB."
        ),
    )
    return parser.parse_args()


def _read_inputs(
    train_tsv: Path, validation_tsv: Path
) -> tuple[list[Locus], set[str]]:
    train_positives = read_positive_loci(train_tsv, split="train")
    validation_positives = read_positive_loci(
        validation_tsv, split="validation"
    )
    train_coordinates = {
        (locus.chromosome, locus.position_1based)
        for locus in train_positives
    }
    validation_coordinates = {
        (locus.chromosome, locus.position_1based)
        for locus in validation_positives
    }
    overlap = train_coordinates & validation_coordinates
    if overlap:
        raise ValueError(
            "Train and validation positive files overlap at coordinates: "
            f"{sorted(overlap)[:10]}"
        )
    validation_chromosomes = {
        locus.chromosome for locus in validation_positives
    }
    train_chromosomes = {locus.chromosome for locus in train_positives}
    chromosome_overlap = train_chromosomes & validation_chromosomes
    if chromosome_overlap:
        raise ValueError(
            "The positive split is not chromosome-held-out; chromosomes occur "
            f"in both inputs: {sorted(chromosome_overlap)}"
        )
    return train_positives + validation_positives, validation_chromosomes


def main() -> int:
    args = parse_args()
    if args.region_length < 2048 or (
        args.region_length & (args.region_length - 1)
    ):
        raise ValueError(
            "--region-length must be an AlphaGenome-compatible power of two "
            "of at least 2048 bp."
        )

    train_tsv = args.train_tsv.expanduser().resolve()
    validation_tsv = args.validation_tsv.expanduser().resolve()
    reference_path = args.reference_fasta.expanduser().resolve()
    output_path = args.output_tsv.expanduser().resolve()
    mask_path = (
        args.mask_bed.expanduser().resolve()
        if args.mask_bed
        else default_mask_path(output_path, args.mask_window)
    )
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )

    positives, validation_chromosomes = _read_inputs(
        train_tsv, validation_tsv
    )
    reference = IndexedFasta(reference_path)
    chromosomes = parse_chromosomes(args.chromosomes, reference.lengths)
    masks = build_aging_mask(
        positives, reference.lengths, args.mask_window
    )
    runs = build_eligible_start_runs(
        reference.lengths,
        masks,
        args.region_length,
        chromosomes,
    )
    eligible_starts = sum(run.count for run in runs)
    masked_bases = sum(mask.end - mask.start for mask in masks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    fieldnames = [
        "sample_id",
        "chromosome",
        "start_0based",
        "end_0based",
        "position_1based",
        "region_length",
        "label",
        "label_type",
        "source",
        "split",
    ]
    if args.include_sequence:
        fieldnames.append("model_sequence")

    stats = SamplingStats()
    chromosome_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    sampler = sample_negative_regions(
        runs,
        number=args.number,
        region_length=args.region_length,
        reference=reference,
        max_n_fraction=args.max_n_fraction,
        max_attempts_per_sample=args.max_attempts_per_sample,
        seed=args.seed,
        include_sequence=args.include_sequence,
        stats=stats,
    )
    with _open_output(output_path, temporary) as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, region in enumerate(sampler, start=1):
            split = (
                "validation"
                if region.chromosome in validation_chromosomes
                else "train"
            )
            row: dict[str, object] = {
                "sample_id": f"NEG_{index:09d}",
                "chromosome": region.chromosome,
                "start_0based": region.start,
                "end_0based": region.end,
                "position_1based": (
                    region.start + args.region_length // 2 + 1
                ),
                "region_length": args.region_length,
                "label": 0,
                "label_type": "random_genome_masked_assumed_negative",
                "source": "hg38_random_after_aging_mask",
                "split": split,
            }
            if args.include_sequence:
                row["model_sequence"] = region.sequence
            writer.writerow(row)
            chromosome_counts[region.chromosome] += 1
            split_counts[split] += 1
            if index % 100_000 == 0 or index == args.number:
                print(
                    f"Wrote {index:,}/{args.number:,} negative regions "
                    f"after {stats.attempted:,} candidate checks."
                )
    temporary.replace(output_path)
    write_mask_bed(mask_path, masks)

    manifest = {
        "inputs": {
            "train": {
                "path": str(train_tsv),
                "sha256": sha256_file(train_tsv),
            },
            "validation": {
                "path": str(validation_tsv),
                "sha256": sha256_file(validation_tsv),
            },
            "positive_rows": len(positives),
        },
        "reference": {
            "path": str(reference_path),
            "assembly": "GRCh38/hg38",
            "chromosomes": {
                chromosome: reference.lengths[chromosome]
                for chromosome in chromosomes
            },
        },
        "parameters": {
            "number": args.number,
            "mask_window": args.mask_window,
            "region_length": args.region_length,
            "max_n_fraction": args.max_n_fraction,
            "max_attempts_per_sample": args.max_attempts_per_sample,
            "seed": args.seed,
            "include_sequence": args.include_sequence,
        },
        "mask": {
            "path": str(mask_path),
            "sha256": sha256_file(mask_path),
            "unmerged_positive_contexts": len(positives),
            "merged_intervals": len(masks),
            "merged_masked_bases": masked_bases,
        },
        "sampling_space": {
            "eligible_unique_region_starts": eligible_starts,
            "negative_regions_may_overlap_each_other": True,
            "duplicate_region_starts": False,
            "full_region_cannot_overlap_aging_mask": True,
        },
        "sampling": {
            "candidate_checks": stats.attempted,
            "rejected_for_ambiguous_sequence": (
                stats.rejected_ambiguous_sequence
            ),
        },
        "split": {
            "strategy": "validation_chromosomes_from_positive_input",
            "validation_chromosomes": sorted(
                validation_chromosomes, key=natural_chromosome_key
            ),
            "counts": dict(sorted(split_counts.items())),
        },
        "output": {
            "path": str(output_path),
            "rows": args.number,
            "counts_by_chromosome": dict(
                sorted(
                    chromosome_counts.items(),
                    key=lambda item: natural_chromosome_key(item[0]),
                )
            ),
        },
        "label_semantics": {
            "0": (
                "random hg38 interval whose full span does not overlap any "
                "known 200 bp aging context; this is an assumed, not "
                "experimentally verified, negative"
            )
        },
        "coordinate_system": (
            "start_0based/end_0based are BED-style half-open coordinates; "
            "position_1based is the first base of the right half of the even "
            "length region"
        ),
    }
    write_manifest(manifest_path, manifest)

    print(f"Dataset:  {output_path}")
    print(f"Mask:     {mask_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
