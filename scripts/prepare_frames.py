#!/usr/bin/env python3
"""Decode all raw AVI frames, then select and zero-base the configured clip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


DEFAULT_CONFIG = Path("configs/s12_sandwich_7150991-2470.json")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def numbered_frames(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jpg"))


def assert_numbering(frames: list[Path], expected_count: int) -> None:
    if len(frames) != expected_count:
        raise ValueError(f"Expected {expected_count} frames; found {len(frames)}")
    for index, frame in enumerate(frames):
        if frame.name != f"{index:07d}.jpg":
            raise ValueError(
                f"Non-contiguous frame sequence at {index}: found {frame.name}"
            )


def decode_all(video_path: Path, raw_dir: Path, expected_count: int) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_frames(raw_dir)
    if existing:
        assert_numbering(existing, expected_count)
        print(f"Using {len(existing)} existing raw frames in {raw_dir}")
        return existing

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(raw_dir / "%07d.jpg"),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required; install FFmpeg first.") from error

    frames = numbered_frames(raw_dir)
    assert_numbering(frames, expected_count)
    print(f"Decoded {len(frames)} exact raw frames into {raw_dir}")
    return frames


def materialize_clip(
    raw_frames: list[Path], clip_dir: Path, start: int, end: int
) -> list[Path]:
    expected_count = end - start
    clip_dir.mkdir(parents=True, exist_ok=True)
    existing = numbered_frames(clip_dir)
    if existing:
        assert_numbering(existing, expected_count)
        print(f"Using {len(existing)} existing clip frames in {clip_dir}")
        return existing

    for local_index, global_index in enumerate(range(start, end)):
        source = raw_frames[global_index]
        target = clip_dir / f"{local_index:07d}.jpg"
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    frames = numbered_frames(clip_dir)
    assert_numbering(frames, expected_count)
    print(
        f"Prepared {len(frames)} local frames: local 0 maps to global {start}"
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = Path(config["video_path"])
    work_dir = Path(config["work_dir"])
    raw_dir = work_dir / "raw_frames"
    clip_dir = work_dir / "frames"
    expected_count = int(config["expected_raw_frames"])
    clip_start = int(config["clip_start_global"])
    clip_end = int(config["clip_end_global"])

    raw_frames = decode_all(video_path, raw_dir, expected_count)
    clip_frames = materialize_clip(raw_frames, clip_dir, clip_start, clip_end)
    shutil.copy2(clip_frames[0], work_dir / "first_frame.jpg")

    metadata = {
        "video_id": config["video_id"],
        "target_object": config["target_object"],
        "clip_start_global": clip_start,
        "clip_end_global": clip_end,
        "num_local_frames": len(clip_frames),
        "fps": config["fps"],
        "local_to_global_rule": "global_frame = clip_start_global + local_frame",
    }
    (work_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Metadata: {work_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
