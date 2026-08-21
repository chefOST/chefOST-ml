#!/usr/bin/env python3
"""Convert raw TubeletGraph VLM JSON into an auditable state-event draft."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/s12_sandwich_7150991-2470.json")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_draft(prediction: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Preserve raw graph descriptions while leaving state fields unmapped."""
    obj_info = prediction.get("obj_info")
    if not isinstance(obj_info, dict):
        raise ValueError("TubeletGraph prediction is missing the obj_info dictionary")

    initial = obj_info.get("0", {})
    if not isinstance(initial, dict):
        raise ValueError("TubeletGraph obj_info['0'] must be a dictionary")

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for object_id, raw_node in obj_info.items():
        if str(object_id) == "0":
            continue
        if not isinstance(raw_node, dict):
            raise ValueError(f"obj_info[{object_id!r}] must be a dictionary")
        if "object_start_frame_idx" not in raw_node:
            raise ValueError(
                f"obj_info[{object_id!r}] has no object_start_frame_idx"
            )
        local_frame = int(raw_node["object_start_frame_idx"])
        by_frame[local_frame].append(
            {
                "tubelet_object_id": str(object_id),
                "description": raw_node.get("desc"),
                "prior_description": raw_node.get("prior_desc"),
                "action": raw_node.get("action"),
                "analysis_frame_idx": raw_node.get("analysis_frame_idx"),
                "object_start_frame_idx": local_frame,
            }
        )

    num_local_frames = int(config["clip_end_global"]) - int(
        config["clip_start_global"]
    )
    transitions = []
    for local_frame, raw_nodes in sorted(by_frame.items()):
        if not 0 <= local_frame < num_local_frames:
            raise ValueError(
                f"Tubelet transition {local_frame} is outside [0, {num_local_frames})"
            )
        transitions.append(
            {
                "local_frame": local_frame,
                "state": None,
                "raw_nodes": sorted(
                    raw_nodes, key=lambda item: item["tubelet_object_id"]
                ),
            }
        )

    return {
        "schema_version": 1,
        "video_id": config["video_id"],
        "object": config["target_object"],
        "clip_start_global": int(config["clip_start_global"]),
        "num_local_frames": num_local_frames,
        "initial_state": None,
        "initial_raw": {
            "tubelet_object_id": "0",
            "description": initial.get("desc"),
        },
        "transitions": transitions,
        "provenance": {
            "source": "TubeletGraph GPT-4.1 prediction JSON",
            "state_mapping_status": "unmapped",
            "ground_truth_used_for_mapping": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    draft = build_draft(load_json(args.prediction), load_json(args.config))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
    print(f"Draft: {args.out}")
    print(f"Candidate transition frames: {len(draft['transitions'])}")
    print("State fields are intentionally null until the fixed vocabulary adapter runs.")


if __name__ == "__main__":
    main()
