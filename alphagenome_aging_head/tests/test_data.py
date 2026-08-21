from __future__ import annotations

from collections import Counter
from io import StringIO
from pathlib import Path
import unittest
from unittest import mock

from aging_alphagenome.data import (
    Locus,
    annotate_positive_sequences,
    centered_bounds,
    choose_validation_chromosomes,
    gc_fraction,
    read_positive_loci,
    sample_matched_negatives,
)


class FakeReference:

    def __init__(self, sequences: dict[str, str]):
        self.sequences = sequences
        self.lengths = {name: len(sequence) for name, sequence in sequences.items()}

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        if start < 0 or end > self.lengths[chromosome]:
            raise ValueError("out of bounds")
        return self.sequences[chromosome][start:end]


class DataTest(unittest.TestCase):

    def test_centered_bounds(self):
        self.assertEqual(centered_bounds(101, 201), (0, 201))
        self.assertEqual(centered_bounds(1025, 2048), (0, 2048))

    def test_gc_fraction_ignores_n(self):
        self.assertAlmostEqual(gc_fraction("ACGTNN"), 0.5)

    def test_coordinate_reader_ignores_all_columns_after_position(self):
        contents = (
            "#CHROM\tPOS\tREF\tALT\tGene.refGene\n"
            "chr1\t3001\tINVALID\tINVALID\tSHOULD_NOT_BE_READ\n"
        )
        with mock.patch.object(Path, "open", return_value=StringIO(contents)):
            loci = read_positive_loci(Path("coordinates.tsv"), split="train")

        self.assertEqual(len(loci), 1)
        locus = loci[0]
        self.assertEqual((locus.chromosome, locus.position_1based), ("chr1", 3001))
        self.assertEqual(locus.label, 1)
        self.assertEqual(locus.split, "train")
        self.assertEqual(locus.reference, "")
        self.assertEqual(locus.alternate, "")
        self.assertEqual(locus.gene, "")

    def test_sequence_annotation_derives_reference_from_hg38(self):
        reference = FakeReference({"chr1": "ACGT" * 2000})
        locus = Locus(
            sample_id="POS_TRAIN_00001",
            pair_id="POS_TRAIN_00001",
            chromosome="chr1",
            position_1based=3001,
            start_0based=3000,
            end_0based=3001,
            reference="",
            alternate="",
            gene="",
            label=1,
            label_type="known_positive",
            source="coordinates.tsv",
            split="train",
        )

        annotate_positive_sequences([locus], reference, 201, 2048)

        self.assertEqual(len(locus.biological_sequence), 201)
        self.assertEqual(len(locus.model_sequence), 2048)
        self.assertEqual(locus.reference, locus.biological_sequence[100])

    def test_chromosome_holdout_is_close_to_twenty_percent(self):
        counts = Counter(
            {
                "chr1": 242,
                "chr2": 186,
                "chr3": 96,
                "chr4": 76,
                "chr5": 95,
                "chr6": 155,
                "chr7": 106,
                "chr8": 66,
                "chr9": 32,
                "chr11": 153,
                "chr12": 104,
                "chr13": 12,
                "chr14": 56,
                "chr15": 79,
                "chr16": 45,
                "chr17": 126,
                "chr18": 14,
                "chr19": 232,
                "chr21": 28,
                "chr22": 104,
                "chrX": 125,
            }
        )
        selected = choose_validation_chromosomes(counts, 0.2, seed=17)
        held_out = sum(counts[chromosome] for chromosome in selected)
        self.assertLessEqual(abs(held_out / sum(counts.values()) - 0.2), 0.001)

    def test_negative_sampling_matches_reference_and_gc(self):
        reference = FakeReference({"chr1": "ACGT" * 6000})
        position = 8001
        biological_start, biological_end = centered_bounds(position, 201)
        biological = reference.fetch("chr1", biological_start, biological_end)
        model_start, model_end = centered_bounds(position, 2048)
        positive = Locus(
            sample_id="POS_1",
            pair_id="POS_1",
            chromosome="chr1",
            position_1based=position,
            start_0based=position - 1,
            end_0based=position,
            reference="A",
            alternate="G",
            gene="GENE",
            label=1,
            label_type="known_positive",
            source="test",
            split="train",
            gc_fraction_biological=gc_fraction(biological),
            biological_sequence=biological,
            model_sequence=reference.fetch("chr1", model_start, model_end),
        )
        negatives = sample_matched_negatives(
            [positive],
            reference,
            biological_window=201,
            model_window=2048,
            negative_ratio=1,
            gc_tolerance=0.01,
            positive_exclusion_radius=2048,
            negative_min_distance=201,
            max_attempts=5000,
            seed=17,
        )
        self.assertEqual(len(negatives), 1)
        negative = negatives[0]
        self.assertEqual(negative.reference, positive.reference)
        self.assertEqual(negative.chromosome, positive.chromosome)
        self.assertEqual(negative.label, 0)
        self.assertEqual(negative.split, "train")
        self.assertLessEqual(
            abs(
                negative.gc_fraction_biological
                - positive.gc_fraction_biological
            ),
            0.01,
        )
        self.assertGreater(
            abs(negative.position_1based - positive.position_1based), 2048
        )


if __name__ == "__main__":
    unittest.main()
