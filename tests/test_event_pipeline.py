from __future__ import annotations

import unittest

from scripts.build_event_draft import build_draft
from scripts.map_event_states import (
    canonical_states,
    map_events,
    normalize_candidate,
    semantic_evidence,
)


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

    def test_object_candidates_are_validated_and_normalized(self) -> None:
        state_dict = {
            "s2i": {"on dish": 0, "placed": 1, "unrelated": 2},
            "all2one": {},
        }
        self.assertEqual(
            canonical_states(state_dict, ["Placed", "on dish"]),
            ["on dish", "placed"],
        )
        with self.assertRaises(ValueError):
            canonical_states(state_dict, ["invented"])

    def test_missing_tubelet_text_abstains_without_model_call(self) -> None:
        class RefusingClient:
            @property
            def chat(self):
                raise AssertionError("Model must not be called without semantic text")

        draft = {
            "object": "bread slices",
            "num_local_frames": 3,
            "initial_raw": {"description": None},
            "transitions": [],
            "provenance": {},
        }
        state_dict = {"s2i": {"on dish": 0}, "all2one": {}, "one2all": {}}
        mapped = map_events(
            draft,
            state_dict,
            RefusingClient(),
            "unused",
            allowed_states=["on dish"],
        )
        self.assertIsNone(mapped["initial_state"])
        self.assertEqual(
            mapped["initial_mapping"]["status"],
            "abstained_no_tubelet_text",
        )
        self.assertEqual(
            mapped["provenance"]["state_mapping_status"],
            "abstained_no_tubelet_text",
        )
        self.assertFalse(mapped["provenance"]["video_frames_used_for_mapping"])

    def test_semantic_evidence_ignores_nonsemantic_metadata(self) -> None:
        self.assertEqual(
            semantic_evidence(
                {
                    "tubelet_object_id": "2",
                    "raw_nodes": [
                        {"description": "bread on a plate", "analysis_frame_idx": 4}
                    ],
                }
            ),
            ["bread on a plate"],
        )


if __name__ == "__main__":
    unittest.main()
