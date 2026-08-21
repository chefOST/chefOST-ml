from __future__ import annotations

import unittest

from scripts.build_event_draft import build_draft
from scripts.map_event_states import normalize_candidate


class EventPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "video_id": "S12_Sandwich_7150991-2470",
            "target_object": "bread slices",
            "clip_start_global": 2824,
            "clip_end_global": 2834,
        }

    def test_draft_consolidates_nodes_at_the_same_frame(self) -> None:
        prediction = {
            "obj_info": {
                "0": {"desc": "bread slices"},
                "2": {
                    "desc": "bread piece",
                    "prior_desc": "bread slice",
                    "action": "spread",
                    "analysis_frame_idx": 5,
                    "object_start_frame_idx": 4,
                },
                "1": {
                    "desc": "bread piece",
                    "prior_desc": "bread slice",
                    "action": "spread",
                    "analysis_frame_idx": 4,
                    "object_start_frame_idx": 4,
                },
            }
        }
        draft = build_draft(prediction, self.config)
        self.assertEqual(draft["initial_state"], None)
        self.assertEqual(len(draft["transitions"]), 1)
        self.assertEqual(draft["transitions"][0]["local_frame"], 4)
        self.assertEqual(
            [
                node["tubelet_object_id"]
                for node in draft["transitions"][0]["raw_nodes"]
            ],
            ["1", "2"],
        )

    def test_candidate_normalizes_known_alias(self) -> None:
        state_dict = {
            "s2i": {"on dish": 0, "topped": 1},
            "all2one": {"on plate": "on dish"},
        }
        self.assertEqual(
            normalize_candidate("on plate", state_dict, ["on dish", "topped"]),
            "on dish",
        )

    def test_candidate_accepts_json_wrapper(self) -> None:
        state_dict = {"s2i": {"on dish": 0}, "all2one": {}}
        self.assertEqual(
            normalize_candidate('{"state": "on dish"}', state_dict, ["on dish"]),
            "on dish",
        )


if __name__ == "__main__":
    unittest.main()
