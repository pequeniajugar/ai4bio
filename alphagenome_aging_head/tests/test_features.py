from __future__ import annotations

import unittest

import numpy as np

from aging_alphagenome.features import one_hot_encode, pooling_weights


class FeatureTest(unittest.TestCase):

    def test_one_hot_and_central_pool_weights(self):
        encoded = one_hot_encode(np.asarray(["ACGTN"], dtype=str))
        np.testing.assert_array_equal(
            encoded[0],
            np.asarray(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
        )
        weights = pooling_weights(2048, 201)
        self.assertEqual(len(weights), 16)
        self.assertAlmostEqual(float(weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
