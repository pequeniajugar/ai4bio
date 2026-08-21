from __future__ import annotations

import unittest

from aging_alphagenome.composition import (
    adjacent_pair_counts,
    base_counts,
    composition_match,
    dinucleotide_counts,
    dinucleotide_shuffle,
    gc_count,
    mononucleotide_shuffle,
    sequence_features,
    verify_dinucleotide_preservation,
)


class CompositionTest(unittest.TestCase):
    def test_sequence_features(self):
        features = sequence_features("AACCGGTT")
        self.assertEqual(features["gc_count"], 4)
        self.assertAlmostEqual(features["gc_fraction"], 0.5)
        self.assertEqual(features["count_A"], 2)
        self.assertEqual(features["count_C"], 2)
        self.assertGreater(features["dinuc_CG"], 0.0)

    def test_mononucleotide_shuffle_is_deterministic_and_preserves_counts(self):
        sequence = "AAAACCCCGGGGTTTTNN"
        first = mononucleotide_shuffle(sequence, seed=17, key="sample")
        second = mononucleotide_shuffle(sequence, seed=17, key="sample")
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(sequence))
        self.assertEqual(base_counts(first), base_counts(sequence))
        self.assertEqual(gc_count(first), gc_count(sequence))


    def test_dinucleotide_shuffle_is_deterministic_and_exact(self):
        sequence = ("AACCGGTTCGATCGGATCCG" * 10) + "N"
        first = dinucleotide_shuffle(sequence, seed=17, key="sample")
        second = dinucleotide_shuffle(sequence, seed=17, key="sample")
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(sequence))
        self.assertEqual(sorted(first), sorted(sequence))
        self.assertEqual(dinucleotide_counts(first), dinucleotide_counts(sequence))
        self.assertEqual(adjacent_pair_counts(first), adjacent_pair_counts(sequence))
        verify_dinucleotide_preservation(sequence, first)

    def test_dinucleotide_shuffle_preserves_cpg_and_all_stride1_pairs(self):
        sequence = "ACCATGCGCGTTAACCGGTTACGCGAT"
        shuffled = dinucleotide_shuffle(sequence, seed=23, key="example")
        original_pairs = adjacent_pair_counts(sequence)
        shuffled_pairs = adjacent_pair_counts(shuffled)
        self.assertEqual(original_pairs, shuffled_pairs)
        self.assertEqual(original_pairs["CG"], shuffled_pairs["CG"])
        self.assertEqual(base_counts(sequence), base_counts(shuffled))

    def test_dinucleotide_shuffle_handles_unique_trail(self):
        # ACCATG has a unique Eulerian trail for its directed pair multigraph,
        # so an exact pair-preserving shuffle is allowed to remain unchanged.
        sequence = "ACCATG"
        shuffled = dinucleotide_shuffle(sequence, seed=41, key="unique")
        self.assertEqual(shuffled, sequence)
        verify_dinucleotide_preservation(sequence, shuffled)

    def test_composition_match_rejects_same_gc_but_different_mono_composition(self):
        target = "A" * 50 + "C" * 50 + "G" * 50 + "T" * 50
        candidate = "A" * 90 + "C" * 10 + "G" * 90 + "T" * 10
        self.assertEqual(gc_count(target), gc_count(candidate))
        self.assertFalse(
            composition_match(
                candidate,
                target,
                gc_count_tolerance=0,
                base_fraction_tolerance=0.05,
                dinucleotide_l1_tolerance=2.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
