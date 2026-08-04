from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import unittest
from unittest import mock

from aging_alphagenome.data import Locus
from aging_alphagenome.negative_data import (
    GenomicInterval,
    SamplingStats,
    build_aging_mask,
    build_eligible_start_runs,
    main,
    merge_intervals,
    sample_negative_regions,
)


class FakeReference:

    def __init__(self, sequences: dict[str, str]):
        self.sequences = sequences
        self.lengths = {
            name: len(sequence) for name, sequence in sequences.items()
        }

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self.sequences[chromosome][start:end]


def positive(chromosome: str, position_1based: int) -> Locus:
    return Locus(
        sample_id=f"POS_{chromosome}_{position_1based}",
        pair_id=f"POS_{chromosome}_{position_1based}",
        chromosome=chromosome,
        position_1based=position_1based,
        start_0based=position_1based - 1,
        end_0based=position_1based,
        reference="",
        alternate="",
        gene="",
        label=1,
        label_type="known_positive",
        source="test",
    )


class NegativeDataTest(unittest.TestCase):

    def test_merge_intervals_merges_overlaps_and_adjacent_intervals(self):
        intervals = [
            GenomicInterval("chr1", 20, 30),
            GenomicInterval("chr1", 10, 20),
            GenomicInterval("chr1", 40, 50),
            GenomicInterval("chr2", 5, 10),
        ]
        self.assertEqual(
            merge_intervals(intervals),
            [
                GenomicInterval("chr1", 10, 30),
                GenomicInterval("chr1", 40, 50),
                GenomicInterval("chr2", 5, 10),
            ],
        )

    def test_even_200bp_mask_is_centered_and_clipped(self):
        masks = build_aging_mask(
            [positive("chr1", 101), positive("chr2", 1)],
            {"chr1": 1000, "chr2": 1000},
            mask_window=200,
        )
        self.assertEqual(
            masks,
            [
                GenomicInterval("chr1", 0, 200),
                GenomicInterval("chr2", 0, 100),
            ],
        )

    def test_eligible_regions_cannot_overlap_mask(self):
        masks = [GenomicInterval("chr1", 20, 30)]
        runs = build_eligible_start_runs(
            {"chr1": 100},
            masks,
            region_length=10,
            chromosomes=["chr1"],
        )
        self.assertEqual(
            runs,
            [
                # Starts 11 through 29 would overlap [20, 30).
                # Thus the valid half-open start runs are [0, 11), [30, 91).
                type(runs[0])("chr1", 0, 11),
                type(runs[0])("chr1", 30, 91),
            ],
        )
        for run in runs:
            for start in range(run.start, run.end):
                self.assertTrue(start + 10 <= 20 or start >= 30)

    def test_sampling_is_reproducible_unique_and_filters_n(self):
        reference = FakeReference(
            {
                "chr1": "A" * 50 + "N" * 20 + "C" * 50,
                "chr2": "G" * 120,
            }
        )
        runs = build_eligible_start_runs(
            reference.lengths,
            [],
            region_length=16,
            chromosomes=["chr1", "chr2"],
        )
        first_stats = SamplingStats()
        first = list(
            sample_negative_regions(
                runs,
                number=40,
                region_length=16,
                reference=reference,
                max_n_fraction=0.0,
                max_attempts_per_sample=20,
                seed=17,
                stats=first_stats,
            )
        )
        second = list(
            sample_negative_regions(
                runs,
                number=40,
                region_length=16,
                reference=reference,
                max_n_fraction=0.0,
                max_attempts_per_sample=20,
                seed=17,
            )
        )
        self.assertEqual(first, second)
        self.assertEqual(
            len({(region.chromosome, region.start) for region in first}), 40
        )
        self.assertGreater(first_stats.rejected_ambiguous_sequence, 0)
        for region in first:
            self.assertNotIn(
                "N",
                reference.fetch(
                    region.chromosome, region.start, region.end
                ),
            )

    def test_main_writes_gzipped_regions_mask_and_manifest(self):
        root = Path(__file__).resolve().parent
        train_tsv = root / ".tmp_negative_train.tsv"
        validation_tsv = root / ".tmp_negative_validation.tsv"
        output = root / ".tmp_negatives.tsv.gz"
        mask = root / ".tmp_mask.bed"
        manifest = root / ".tmp_manifest.json"
        temporary_outputs = [
            train_tsv,
            validation_tsv,
            output,
            mask,
            manifest,
            output.with_suffix(output.suffix + ".tmp"),
            mask.with_suffix(mask.suffix + ".tmp"),
            manifest.with_suffix(manifest.suffix + ".tmp"),
        ]
        try:
            train_tsv.write_text(
                "#CHROM\tPOS\nchr1\t5001\n", encoding="utf-8"
            )
            validation_tsv.write_text(
                "#CHROM\tPOS\nchr2\t5001\n", encoding="utf-8"
            )
            arguments = argparse.Namespace(
                train_tsv=train_tsv,
                validation_tsv=validation_tsv,
                reference_fasta=root / "reference.fa",
                output_tsv=output,
                mask_bed=mask,
                manifest=manifest,
                number=20,
                mask_window=200,
                region_length=2048,
                chromosomes=None,
                max_n_fraction=0.0,
                max_attempts_per_sample=20,
                seed=17,
                include_sequence=False,
            )
            reference = FakeReference(
                {"chr1": "A" * 20_000, "chr2": "C" * 20_000}
            )

            with (
                mock.patch(
                    "aging_alphagenome.negative_data.parse_args",
                    return_value=arguments,
                ),
                mock.patch(
                    "aging_alphagenome.negative_data.IndexedFasta",
                    return_value=reference,
                ),
            ):
                self.assertEqual(main(), 0)

            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 20)
            self.assertEqual(
                {
                    int(row["end_0based"]) - int(row["start_0based"])
                    for row in rows
                },
                {2048},
            )
            self.assertEqual(
                {
                    row["split"]
                    for row in rows
                    if row["chromosome"] == "chr2"
                },
                {"validation"},
            )
            self.assertNotIn("model_sequence", rows[0])

            mask_rows = mask.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(mask_rows), 2)
            self.assertTrue(mask_rows[0].startswith("chr1\t"))
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(metadata["output"]["rows"], 20)
            self.assertTrue(
                metadata["sampling_space"][
                    "full_region_cannot_overlap_aging_mask"
                ]
            )
        finally:
            for path in temporary_outputs:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
