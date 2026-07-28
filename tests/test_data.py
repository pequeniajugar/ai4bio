from __future__ import annotations

from collections import Counter
import unittest

from aging_alphagenome.data import (
    Locus,
    centered_bounds,
    choose_validation_chromosomes,
    gc_fraction,
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
