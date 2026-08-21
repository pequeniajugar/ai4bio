import csv
import json
import tempfile
import unittest
from pathlib import Path

from aging_alphagenome.negative_sampling_transfer_summary import (
    _extract_audit_row,
    _score_metrics,
    verify_positive_identity,
)


class NegativeSamplingTransferSummaryTest(unittest.TestCase):
    def test_positive_identity_accepts_same_positives_with_different_negatives(self):
        matched = [
            {
                "sample_id": "POS_1",
                "label": "1",
                "chromosome": "chr1",
                "position_1based": "101",
                "split": "validation",
                "biological_sequence": "ACG",
            },
            {
                "sample_id": "NEG_1",
                "label": "0",
                "chromosome": "chr1",
                "position_1based": "501",
                "split": "validation",
                "biological_sequence": "AAA",
            },
        ]
        unmatched = [
            dict(matched[0]),
            {
                "sample_id": "NEG_1",
                "label": "0",
                "chromosome": "chr1",
                "position_1based": "901",
                "split": "validation",
                "biological_sequence": "CCC",
            },
        ]
        result = verify_positive_identity(matched, unmatched)
        self.assertEqual(result["positive_count"], 1)
        self.assertTrue(result["identical_positive_ids_coordinates_splits_sequences"])

    def test_positive_identity_rejects_changed_positive_sequence(self):
        matched = [{
            "sample_id": "POS_1", "label": "1", "chromosome": "chr1",
            "position_1based": "101", "split": "validation",
            "biological_sequence": "ACG",
        }]
        unmatched = [dict(matched[0], biological_sequence="ACT")]
        with self.assertRaises(ValueError):
            verify_positive_identity(matched, unmatched)

    def test_score_metrics_supports_score_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            metrics = {
                "auroc": 0.7,
                "average_precision": 0.6,
                "accuracy_at_0.5": 0.55,
                "balanced_accuracy_at_0.5": 0.56,
                "loss": 0.8,
            }
            path.write_text(json.dumps({"metrics": metrics}))
            observed, _ = _score_metrics(path)
            self.assertEqual(observed, metrics)

    def test_extract_audit_row(self):
        payload = {
            "distribution_comparisons": {
                "validation": {
                    "gc_fraction": {
                        "positive": {"mean": 0.5},
                        "negative": {"mean": 0.4},
                        "positive_minus_negative_mean": 0.1,
                        "cohen_d": 1.0,
                        "ks_statistic": 0.2,
                        "univariate_auroc": 0.8,
                    }
                }
            },
            "composition_only_baselines": {
                name: {"test_metrics": {"auroc": 0.75, "accuracy": 0.7}}
                for name in (
                    "gc_only",
                    "mononucleotide",
                    "mono_plus_cpg",
                    "mono_plus_dinucleotide",
                )
            },
            "paired_matching": {"mean_absolute_gc_fraction_delta": 0.03},
        }
        row = _extract_audit_row("x", payload)
        self.assertAlmostEqual(row["positive_minus_negative_gc"], 0.1)
        self.assertAlmostEqual(row["gc_only_auroc"], 0.75)
        self.assertAlmostEqual(row["mean_absolute_paired_gc_delta"], 0.03)


if __name__ == "__main__":
    unittest.main()
