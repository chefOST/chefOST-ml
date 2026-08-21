from __future__ import annotations

import unittest

from scripts.evaluate import (
    densify,
    evaluate_events,
    f1_threshold_curve,
    normalize_state,
)


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state_dict = {
            "s2i": {"on dish": 0, "topped": 1, "assembled": 2},
            "all2one": {"on plate": "on dish"},
        }

    def test_densify_forward_fills_transitions(self) -> None:
        events = {
            "num_local_frames": 6,
            "initial_state": "on plate",
            "transitions": [
                {"local_frame": 2, "state": "topped"},
                {"local_frame": 5, "state": "assembled"},
            ],
        }
        self.assertEqual(
            densify(events, self.state_dict),
            ["on dish", "on dish", "topped", "topped", "topped", "assembled"],
        )

    def test_normalize_rejects_unknown_state(self) -> None:
        with self.assertRaises(ValueError):
            normalize_state("invented", self.state_dict)

    def test_evaluation_skips_empty_and_accepts_multilabel_hit(self) -> None:
        annotations = {
            "video": {
                "object": [[], [], [], [], []],
                "state": {
                    "bread": [
                        [],
                        ["on dish"],
                        ["on dish", "topped"],
                        ["assembled"],
                        [],
                    ]
                },
            }
        }
        events = {
            "video_id": "video",
            "object": "bread",
            "clip_start_global": 1,
            "num_local_frames": 3,
            "initial_state": "on dish",
            "transitions": [{"local_frame": 1, "state": "topped"}],
        }
        rows, summary = evaluate_events(annotations, self.state_dict, events)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary["evaluated_frames"], 3)
        self.assertEqual(summary["correct_frames"], 2)
        self.assertAlmostEqual(summary["frame_hit_accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["precision"], summary["micro_precision"])
        self.assertEqual(summary["f1"], summary["micro_f1"])
        self.assertEqual(summary["f1_max"], summary["micro_f1"])
        self.assertEqual(summary["f1_max_threshold"], 0.1)
        self.assertEqual(len(summary["f1_threshold_curve"]), 9)

    def test_f1_max_uses_best_global_threshold(self) -> None:
        curve, best = f1_threshold_curve(
            [[1, 0], [0, 1]],
            [[0.8, 0.2], [0.4, 0.6]],
            thresholds=(0.3, 0.5, 0.7),
        )
        self.assertEqual(len(curve), 3)
        self.assertEqual(best["threshold"], 0.5)
        self.assertEqual(best["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
