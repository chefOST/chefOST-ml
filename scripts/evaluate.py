#!/usr/bin/env python3
"""Densify TubeletGraph state events and compare them with MOSCATO per frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import precision_recall_fscore_support


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_states(state_dict: dict[str, Any]) -> list[str]:
    return sorted(state.strip().lower() for state in state_dict["s2i"])


def normalize_state(state: str, state_dict: dict[str, Any]) -> str:
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"State must be a non-empty string; got {state!r}")
    normalized = state.strip().lower()
    aliases = {
        key.strip().lower(): value.strip().lower()
        for key, value in state_dict.get("all2one", {}).items()
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in set(canonical_states(state_dict)):
        raise ValueError(
            f"State {state!r} is not a canonical MOSCATO state or known alias"
        )
    return normalized


def densify(events: dict[str, Any], state_dict: dict[str, Any]) -> list[str]:
    frame_count = int(events["num_local_frames"])
    if frame_count <= 0:
        raise ValueError("num_local_frames must be positive")

    current_state = normalize_state(events.get("initial_state"), state_dict)
    dense: list[str] = []
    cursor = 0
    previous_transition = -1
    transitions = sorted(
        events.get("transitions", []), key=lambda item: int(item["local_frame"])
    )
    for transition in transitions:
        frame = int(transition["local_frame"])
        if not 0 <= frame < frame_count:
            raise ValueError(f"Transition frame {frame} is outside [0, {frame_count})")
        if frame == previous_transition:
            raise ValueError(f"Duplicate transition frame: {frame}")
        dense.extend([current_state] * (frame - cursor))
        current_state = normalize_state(transition.get("state"), state_dict)
        cursor = frame
        previous_transition = frame
    dense.extend([current_state] * (frame_count - cursor))
    if len(dense) != frame_count:
        raise AssertionError("Densification produced the wrong number of frames")
    return dense


def evaluate_events(
    annotations: dict[str, Any],
    state_dict: dict[str, Any],
    events: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_id = events["video_id"]
    object_name = events["object"]
    clip_start = int(events.get("clip_start_global", 0))
    if video_id not in annotations:
        raise KeyError(f"{video_id} not found in MOSCATO annotations")
    video_gt = annotations[video_id]
    if object_name not in video_gt["state"]:
        raise KeyError(
            f"{object_name!r} is unavailable; choose from {sorted(video_gt['state'])}"
        )

    gt_timeline = video_gt["state"][object_name]
    predictions = densify(events, state_dict)
    if clip_start < 0 or clip_start >= len(gt_timeline):
        raise ValueError(f"clip_start_global {clip_start} is outside the GT timeline")
    if clip_start + len(predictions) > len(gt_timeline):
        raise ValueError(
            "Predicted clip extends beyond MOSCATO: "
            f"{clip_start} + {len(predictions)} > {len(gt_timeline)}"
        )

    canonical = canonical_states(state_dict)
    label_to_idx = {label: index for index, label in enumerate(canonical)}
    rows: list[dict[str, Any]] = []
    y_true: list[list[int]] = []
    y_pred: list[list[int]] = []
    hits = 0

    for local_frame, prediction in enumerate(predictions):
        global_frame = clip_start + local_frame
        gt_states = [
            normalize_state(state, state_dict)
            for state in gt_timeline[global_frame]
        ]
        if not gt_states:
            rows.append(
                {
                    "local_frame": local_frame,
                    "global_frame": global_frame,
                    "prediction": prediction,
                    "gt_states": "",
                    "evaluated": 0,
                    "correct": "",
                }
            )
            continue

        correct = prediction in gt_states
        hits += int(correct)
        true_vector = [0] * len(canonical)
        pred_vector = [0] * len(canonical)
        for gt_state in gt_states:
            true_vector[label_to_idx[gt_state]] = 1
        pred_vector[label_to_idx[prediction]] = 1
        y_true.append(true_vector)
        y_pred.append(pred_vector)
        rows.append(
            {
                "local_frame": local_frame,
                "global_frame": global_frame,
                "prediction": prediction,
                "gt_states": "|".join(gt_states),
                "evaluated": 1,
                "correct": int(correct),
            }
        )

    evaluated = len(y_true)
    if evaluated == 0:
        raise RuntimeError("No annotated frames were evaluated")
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    summary = {
        "video_id": video_id,
        "object": object_name,
        "clip_start_global": clip_start,
        "predicted_frames": len(predictions),
        "evaluated_frames": evaluated,
        "skipped_empty_gt_frames": len(predictions) - evaluated,
        "correct_frames": hits,
        "frame_hit_accuracy": hits / evaluated,
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "policy": {
            "multi_label": "prediction is correct if present in GT state list",
            "empty_gt": "skip",
            "temporal": "initial propagation plus transition forward fill",
        },
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "local_frame",
                "global_frame",
                "prediction",
                "gt_states",
                "evaluated",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path)
    args = parser.parse_args()

    rows, summary = evaluate_events(
        load_json(args.annotations),
        load_json(args.state_dict),
        load_json(args.events),
    )
    write_csv(args.out_csv, rows)
    summary_path = args.out_summary or args.out_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Comparison CSV: {args.out_csv}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
