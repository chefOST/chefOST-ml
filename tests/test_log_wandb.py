from __future__ import annotations

import unittest

from scripts.log_wandb import metric_payload


class WandbLoggingTests(unittest.TestCase):
    def test_primary_metrics_use_word_accuracy_namespace(self) -> None:
        metrics = metric_payload(
            {
                "precision": 0.7,
                "accuracy": 0.6,
                "f1": 0.65,
                "f1_max": 0.8,
                "f1_max_threshold": 0.4,
                "evaluated_frames": 10,
            }
        )
        self.assertEqual(
            set(metrics),
            {
                "word_accuracy/precision",
                "word_accuracy/accuracy",
                "word_accuracy/f1",
                "word_accuracy/f1_max",
                "word_accuracy/f1_max_threshold",
                "word_accuracy/evaluated_frames",
            },
        )

    def test_missing_primary_metric_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            metric_payload({"precision": 1.0})


if __name__ == "__main__":
    unittest.main()
