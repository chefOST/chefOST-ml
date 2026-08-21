#!/usr/bin/env python3
"""Validate the exact CMU video against its MOSCATO annotation timeline."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_CONFIG = Path("configs/s12_sandwich_7150991-2470.json")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,r_frame_rate,nb_frames,width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required; install FFmpeg first.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip() or "ffprobe failed") from error

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}; found {len(streams)}")
    return streams[0]


def validate(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    video_path = Path(config["video_path"])
    annotations_path = Path(config["annotations_path"])
    files_dict_path = Path(config["files_dict_path"])
    video_id = config["video_id"]
    target_object = config["target_object"]

    for required in (video_path, annotations_path, files_dict_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    if video_path.stem != video_id:
        raise ValueError(
            f"AVI stem {video_path.stem!r} does not match video_id {video_id!r}"
        )

    annotations = load_json(annotations_path)
    files_dict = load_json(files_dict_path)
    if video_id not in annotations:
        raise KeyError(f"{video_id} is missing from {annotations_path}")
    if video_id not in files_dict:
        raise KeyError(f"{video_id} is missing from {files_dict_path}")

    video_gt = annotations[video_id]
    object_frames = len(video_gt["object"])
    state_lengths = {
        object_name: len(timeline)
        for object_name, timeline in video_gt["state"].items()
    }
    mismatched = {
        object_name: length
        for object_name, length in state_lengths.items()
        if length != object_frames
    }
    if mismatched:
        raise ValueError(
            f"State timelines do not match object timeline ({object_frames}): {mismatched}"
        )
    if target_object not in video_gt["state"]:
        raise KeyError(
            f"Target {target_object!r} is unavailable; "
            f"choose from {sorted(video_gt['state'])}"
        )

    stream = probe_video(video_path)
    raw_frames = int(stream["nb_frames"])
    fps = Fraction(stream["avg_frame_rate"])
    checks = {
        "raw_frames_equal_gt": raw_frames == object_frames,
        "raw_frames_equal_manifest": raw_frames == int(config["expected_raw_frames"]),
        "fps_equal_manifest": fps == Fraction(int(config["fps"]), 1),
        "dimensions_equal_manifest": (
            int(stream["width"]) == int(config["width"])
            and int(stream["height"]) == int(config["height"])
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Alignment validation failed: {', '.join(failed)}")

    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])
    if not 0 <= clip_start < clip_end <= raw_frames:
        raise ValueError(
            f"Invalid clip [{clip_start}, {clip_end}) for {raw_frames} raw frames"
        )

    original_path = files_dict[video_id].get("video_path", "")
    if Path(original_path).stem != video_id:
        raise ValueError(
            f"MOSCATO files_dict points at an unexpected video: {original_path!r}"
        )

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "moscato_original_video_path": original_path,
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps),
        "raw_frames": raw_frames,
        "moscato_frames": object_frames,
        "target_object": target_object,
        "available_objects": sorted(video_gt["state"]),
        "clip_start_global": clip_start,
        "clip_end_global": clip_end,
        "num_local_frames": clip_end - clip_start,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, help="Optional metadata JSON output")
    args = parser.parse_args()

    report = validate(args.config)
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
