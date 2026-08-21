from __future__ import annotations

import unittest

import numpy as np

from aging_alphagenome.head import (
    binary_auroc,
    classification_metrics,
    fit_logistic_head,
    sigmoid,
)


class HeadTest(unittest.TestCase):

    def test_classification_metrics_rejects_non_finite_scores(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            classification_metrics(
                np.asarray([0, 1], dtype=np.int8),
                np.asarray([0.25, np.nan], dtype=np.float64),
            )

    def test_auroc(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(binary_auroc(labels, scores), 1.0)

    def test_logistic_head_learns_separable_features(self):
        rng = np.random.default_rng(7)
        negative = rng.normal(-1.0, 0.3, size=(80, 8))
        positive = rng.normal(1.0, 0.3, size=(80, 8))
        features = np.concatenate([negative, positive]).astype(np.float32)
        labels = np.concatenate(
            [np.zeros(80, dtype=np.int8), np.ones(80, dtype=np.int8)]
        )
        order = rng.permutation(len(features))
        features = features[order]
        labels = labels[order]

        weights, bias, mean, std, _ = fit_logistic_head(
            features[:120],
            labels[:120],
            features[120:],
            labels[120:],
            learning_rate=0.01,
            l2=1e-4,
            epochs=100,
            batch_size=32,
            patience=15,
            seed=11,
        )
        scores = sigmoid(((features[120:] - mean) / std) @ weights + bias)
        self.assertGreater(binary_auroc(labels[120:], scores), 0.98)


if __name__ == "__main__":
    unittest.main()
