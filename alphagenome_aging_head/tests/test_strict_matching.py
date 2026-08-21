from __future__ import annotations

import unittest

from aging_alphagenome.composition import gc_count
from aging_alphagenome.data import Locus, centered_bounds, gc_fraction, sample_matched_negatives


class FakeReference:
    def __init__(self, sequence: str):
        self.sequences = {"chr1": sequence}
        self.lengths = {"chr1": len(sequence)}

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        return self.sequences[chromosome][start:end]


class StrictMatchingTest(unittest.TestCase):
    def test_gc_exact_matches_integer_gc_count(self):
        reference = FakeReference("ACGT" * 10000)
        position = 12001
        bio_start, bio_end = centered_bounds(position, 201)
        biological = reference.fetch("chr1", bio_start, bio_end)
        model_start, model_end = centered_bounds(position, 2048)
        positive = Locus(
            sample_id="POS_1",
            pair_id="POS_1",
            chromosome="chr1",
            position_1based=position,
            start_0based=position - 1,
            end_0based=position,
            reference=biological[100],
            alternate="",
            gene="",
            label=1,
            label_type="known_positive",
            source="test",
            split="train",
            gc_fraction_biological=gc_fraction(biological),
            biological_sequence=biological,
            model_sequence=reference.fetch("chr1", model_start, model_end),
        )
        negative = sample_matched_negatives(
            [positive],
            reference,
            biological_window=201,
            model_window=2048,
            negative_ratio=1,
            gc_tolerance=0.05,
            positive_exclusion_radius=2048,
            negative_min_distance=201,
            max_attempts=10000,
            seed=17,
            matching_strategy="gc_exact",
            gc_count_tolerance=0,
        )[0]
        self.assertEqual(gc_count(negative.biological_sequence), gc_count(positive.biological_sequence))


if __name__ == "__main__":
    unittest.main()
