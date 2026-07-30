from __future__ import annotations

import unittest

import numpy as np

from aging_alphagenome.cnn import (
    architecture_parameter_count,
    make_development_masks,
    padded_batches,
    reverse_complement,
)
from aging_alphagenome.features import one_hot_encode


class CnnTest(unittest.TestCase):

    def test_architecture_parameter_counts(self):
        self.assertEqual(architecture_parameter_count("small"), 141_601)
        self.assertEqual(architecture_parameter_count("large"), 3_762_305)

    def test_reverse_complement_is_correct_and_reversible(self):
        encoded = one_hot_encode(np.asarray(["ACGTN"], dtype=str))
        reversed_once = reverse_complement(encoded)
        expected = one_hot_encode(np.asarray(["NACGT"], dtype=str))
        np.testing.assert_array_equal(reversed_once, expected)
        np.testing.assert_array_equal(
            reverse_complement(reversed_once), encoded
        )

    def test_internal_validation_keeps_pairs_together(self):
        pair_ids = np.asarray(
            ["p1", "p1", "p2", "p2", "p3", "p3", "t1", "t1"],
            dtype=str,
        )
        splits = np.asarray(
            ["train"] * 6 + ["validation"] * 2,
            dtype=str,
        )
        train, internal_validation, test = make_development_masks(
            splits,
            pair_ids,
            validation_fraction=0.25,
            seed=17,
        )

        self.assertFalse(np.any(train & internal_validation))
        np.testing.assert_array_equal(test, splits == "validation")
        for pair_id in ("p1", "p2", "p3"):
            indices = pair_ids == pair_id
            self.assertTrue(
                np.all(train[indices]) or np.all(internal_validation[indices])
            )

    def test_padded_batches_masks_padding(self):
        batches = list(padded_batches(np.arange(5), batch_size=4))
        self.assertEqual(len(batches), 2)
        np.testing.assert_array_equal(batches[1][0], [4, 4, 4, 4])
        np.testing.assert_array_equal(batches[1][1], [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
