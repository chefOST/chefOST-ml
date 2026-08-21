"""Run pinned TubeletGraph inference for the CMU S12 MOSCATO experiment on Modal.

One-time local setup:
    python3 -m pip install modal
    modal setup
    modal volume create tubelet-data
    modal secret create tubelet-openai OPENAI_API_KEY="$OPENAI_API_KEY"

Smoke test the official bundled example first:
    modal run modal_tubelet.py --smoke-test

Run the prepared S12 clip:
    modal run modal_tubelet.py \
        --video-id S12_Sandwich_7150991-2470 \
        --fps 30 \
        --clip-start-global 2824 \
        --object-name "bread slices"
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

import modal


APP_NAME = "tubeletgraph-moscato-cmu"
VOLUME_NAME = "tubelet-data"
SECRET_NAME = "tubelet-openai"
TUBELETGRAPH_REPOSITORY = "https://github.com/YihongSun/TubeletGraph.git"
TUBELETGRAPH_COMMIT = "fdb05b6fbd7f4644aea990bf967cc18d82bf291b"
DETECTRON2_COMMIT = "a2f4a8771ab77e8411c26b27f24f9489a28a2453"
DEFAULT_VIDEO_ID = "S12_Sandwich_7150991-2470"
DEFAULT_OBJECT = "bread slices"
DEFAULT_CLIP_START = 2824
DEFAULT_FPS = 30
DEFAULT_OBJECT_ID = 1
DEFAULT_EXPECTED_FRAMES = 3890


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
openai_secret = modal.Secret.from_name(
    SECRET_NAME, required_keys=["OPENAI_API_KEY"]
)


tubelet_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libjpeg-dev",
        "libpng-dev",
        "ninja-build",
        "wget",
    )
    .env(
        {
            "FORCE_CUDA": "1",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .run_commands(
        f"git clone --recurse-submodules {TUBELETGRAPH_REPOSITORY} /opt/TubeletGraph",
        f"cd /opt/TubeletGraph && git checkout {TUBELETGRAPH_COMMIT} && "
        "git submodule update --init --recursive",
        "cd /opt/TubeletGraph && python -m pip install "
        "torch==2.7.0 torchvision==0.22.0 "
        "--index-url https://download.pytorch.org/whl/cu126",
        "python -m pip install --upgrade "
        "pip==25.3 setuptools==80.9.0 wheel",
        "cd /opt/TubeletGraph && "
        "grep -v '^mmcv==' requirements.txt > /tmp/tubelet-requirements.txt && "
        "python -m pip install -r /tmp/tubelet-requirements.txt",
        "cd /opt/TubeletGraph && "
        "python -m pip install mmcv==2.2.0 --no-build-isolation",
        "cd /opt/TubeletGraph && bash thirdparty/setup_ckpts.sh",
        "cd /opt/TubeletGraph/thirdparty/sam2 && python -m pip install -e .",
        "cd /opt/TubeletGraph/thirdparty/sam2 && "
        'python -m pip install -e ".[notebooks]"',
        "cd /opt/TubeletGraph/thirdparty/sam2 && "
        "python setup.py build_ext --inplace",
        "cd /opt/TubeletGraph/thirdparty && "
        "git clone https://github.com/facebookresearch/detectron2.git",
        f"cd /opt/TubeletGraph/thirdparty/detectron2 && "
        f"git checkout {DETECTRON2_COMMIT}",
        "cd /opt/TubeletGraph/thirdparty && "
        "python -m pip install -e detectron2 --no-build-isolation",
        "ln -s /opt/TubeletGraph/thirdparty/Entity/Entityv2/CropFormer "
        "/opt/TubeletGraph/thirdparty/detectron2/projects/CropFormer",
        "cd /opt/TubeletGraph/thirdparty/detectron2/projects/"
        "CropFormer/mask2former/modeling/pixel_decoder/ops && bash make.sh",
        "cd /opt/TubeletGraph/thirdparty/fc-clip && "
        "python -m pip install -r requirements.txt",
    )
    .add_local_dir(
        "scripts",
        "/opt/moscato/scripts",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def _run_name(value: str) -> str:
    if value:
        return _safe_component(value, "run_name")
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@app.function(
    image=tubelet_image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=32768,
    ephemeral_disk=100 * 1024,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume},
    secrets=[openai_secret],
    single_use_containers=True,
)
def run_tubelet(
    video_id: str = DEFAULT_VIDEO_ID,
    fps: int = DEFAULT_FPS,
    clip_start_global: int = DEFAULT_CLIP_START,
    object_name: str = DEFAULT_OBJECT,
    object_id: int = DEFAULT_OBJECT_ID,
    expected_frames: int = DEFAULT_EXPECTED_FRAMES,
    run_name: str = "",
    smoke_test: bool = False,
    persist_intermediates: bool = False,
) -> dict:
    """Validate inputs, run TubeletGraph, map states, and persist artifacts."""
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import traceback

    import numpy as np
    from PIL import Image
    import torch

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not injected by the Modal Secret")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run TubeletGraph")
    if fps <= 0:
        raise ValueError("fps must be positive")

    video_id = _safe_component(video_id, "video_id")
    resolved_run_name = _run_name(run_name)
    repository = Path("/opt/TubeletGraph")
    started_at = datetime.now(timezone.utc).isoformat()

    if smoke_test:
        logical_video_id = "0334_cut_fruit_1"
        input_frames = repository / "assets/example/0334_cut_fruit_1"
        input_mask = repository / "assets/example/0334_cut_fruit_1_0000000.png"
        run_dir = Path("/data/smoke_tests") / resolved_run_name
        expected_input_frames = None
        state_dict_path = None
    else:
        logical_video_id = video_id
        input_root = Path("/data") / video_id
        input_frames = input_root / "frames"
        input_mask = input_root / "mask.png"
        state_dict_path = input_root / "state_dict.json"
        run_dir = input_root / "runs" / resolved_run_name
        expected_input_frames = int(expected_frames)

    if run_dir.exists():
        raise FileExistsError(
            f"Run output already exists: {run_dir}. Choose a different --run-name."
        )
    run_dir.mkdir(parents=True)
    log_path = run_dir / "tubeletgraph.log"
    manifest_path = run_dir / "run_manifest.json"

    manifest = {
        "status": "starting",
        "started_at": started_at,
        "run_name": resolved_run_name,
        "smoke_test": smoke_test,
        "video_id": logical_video_id,
        "fps": fps,
        "clip_start_global": None if smoke_test else int(clip_start_global),
        "target_object": None if smoke_test else object_name,
        "target_object_id": None if smoke_test else int(object_id),
        "expected_local_frames": expected_input_frames,
        "tubeletgraph_repository": TUBELETGRAPH_REPOSITORY,
        "tubeletgraph_commit": TUBELETGRAPH_COMMIT,
        "detectron2_commit": DETECTRON2_COMMIT,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3
        ),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "openai_secret_present": True,
        "ground_truth_available_to_gpu_job": False,
        "artifacts": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    caught_error: Exception | None = None
    try:
        if not input_frames.is_dir():
            raise FileNotFoundError(input_frames)
        if not input_mask.is_file():
            raise FileNotFoundError(input_mask)
        if not smoke_test and (state_dict_path is None or not state_dict_path.is_file()):
            raise FileNotFoundError(
                f"{state_dict_path} is required for closed-vocabulary state mapping"
            )

        entries = sorted(path for path in input_frames.iterdir() if path.is_file())
        non_jpegs = [path.name for path in entries if path.suffix.lower() != ".jpg"]
        if non_jpegs:
            raise ValueError(f"Frame directory contains non-JPEG files: {non_jpegs[:5]}")
        frame_paths = [path for path in entries if path.suffix.lower() == ".jpg"]
        if not frame_paths:
            raise ValueError(f"No JPEG frames found in {input_frames}")
        if expected_input_frames is not None and len(frame_paths) != expected_input_frames:
            raise ValueError(
                f"Expected {expected_input_frames} frames; found {len(frame_paths)}"
            )
        for index, frame_path in enumerate(frame_paths):
            if frame_path.name != f"{index:07d}.jpg":
                raise ValueError(
                    f"Frame numbering breaks at {index}: found {frame_path.name}"
                )

        first_frame = Image.open(frame_paths[0])
        mask = np.asarray(Image.open(input_mask))
        if mask.ndim != 2:
            raise ValueError(f"Mask must be single channel; shape is {mask.shape}")
        if mask.shape != (first_frame.height, first_frame.width):
            raise ValueError(
                f"Mask shape {mask.shape} != first frame "
                f"{(first_frame.height, first_frame.width)}"
            )
        mask_values = sorted(int(value) for value in np.unique(mask))
        if smoke_test:
            if not set(mask_values).issubset(set(range(256))):
                raise ValueError(f"Unexpected smoke-test mask values: {mask_values}")
        elif mask_values != [0, int(object_id)]:
            raise ValueError(
                f"Expected mask values [0, {object_id}]; found {mask_values}"
            )

        # Copy Volume inputs to local SSD. Naming the directory after video_id is
        # essential because TubeletGraph derives its output instance name from it.
        local_input_root = Path("/tmp/tubelet_input")
        local_frames = local_input_root / logical_video_id
        local_mask = local_input_root / f"{logical_video_id}_0000000.png"
        local_input_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(input_frames, local_frames)
        shutil.copy2(input_mask, local_mask)

        command = [
            "python",
            "quick_run.py",
            "--input_dir",
            str(local_frames),
            "--input_mask",
            str(local_mask),
            "--fps",
            str(fps),
        ]
        manifest["input_frame_count"] = len(frame_paths)
        manifest["input_dimensions"] = [first_frame.width, first_frame.height]
        manifest["mask_values"] = mask_values
        manifest["command"] = command
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=repository,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        if not smoke_test:
            prediction_json = (
                repository
                / "_pred_out"
                / f"custom-{video_id}-Ours_gpt-4.1"
                / f"{video_id}_{object_id}.json"
            )
            if not prediction_json.is_file():
                raise FileNotFoundError(
                    f"Expected TubeletGraph VLM prediction was not produced: {prediction_json}"
                )

            sys.path.insert(0, "/opt/moscato")
            from openai import OpenAI
            from scripts.build_event_draft import build_draft
            from scripts.map_event_states import map_events

            prediction = json.loads(prediction_json.read_text(encoding="utf-8"))
            state_dict = json.loads(state_dict_path.read_text(encoding="utf-8"))
            event_config = {
                "video_id": video_id,
                "target_object": object_name,
                "clip_start_global": int(clip_start_global),
                "clip_end_global": int(clip_start_global) + len(frame_paths),
            }
            draft = build_draft(prediction, event_config)
            draft_path = run_dir / "tubelet_events.draft.json"
            draft_path.write_text(json.dumps(draft, indent=2) + "\n")
            events = map_events(
                draft,
                state_dict,
                local_frames,
                OpenAI(),
                "gpt-4.1",
            )
            events_path = run_dir / "tubelet_events.json"
            events_path.write_text(json.dumps(events, indent=2) + "\n")
            manifest["artifacts"]["event_draft"] = draft_path.name
            manifest["artifacts"]["mapped_events"] = events_path.name

        manifest["status"] = "succeeded"
    except Exception as error:  # Persist partial outputs before re-raising.
        caught_error = error
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
    finally:
        artifact_errors = []
        artifact_sources = {
            "predictions": repository / "_pred_out",
            "visualizations": repository / "_vis_out",
        }
        if persist_intermediates:
            artifact_sources["intermediates"] = repository / "_interm_out"
        for artifact_name, source in artifact_sources.items():
            if not source.exists():
                continue
            destination = run_dir / artifact_name
            try:
                shutil.copytree(source, destination)
                manifest["artifacts"][artifact_name] = artifact_name
            except Exception as artifact_error:
                artifact_errors.append(
                    f"{artifact_name}: {type(artifact_error).__name__}: {artifact_error}"
                )
        custom_configs = sorted((repository / "configs").glob("custom_*.yaml"))
        if custom_configs:
            config_dir = run_dir / "generated_configs"
            config_dir.mkdir(exist_ok=True)
            for config_path in custom_configs:
                shutil.copy2(config_path, config_dir / config_path.name)
            manifest["artifacts"]["generated_configs"] = "generated_configs"
        if artifact_errors:
            manifest["artifact_errors"] = artifact_errors
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["volume_path"] = str(run_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        data_volume.commit()

    if caught_error is not None:
        raise caught_error
    return {
        "status": manifest["status"],
        "volume": VOLUME_NAME,
        "volume_path": str(run_dir),
        "manifest": str(manifest_path),
    }


@app.local_entrypoint()
def main(
    video_id: str = DEFAULT_VIDEO_ID,
    fps: int = DEFAULT_FPS,
    clip_start_global: int = DEFAULT_CLIP_START,
    object_name: str = DEFAULT_OBJECT,
    object_id: int = DEFAULT_OBJECT_ID,
    expected_frames: int = DEFAULT_EXPECTED_FRAMES,
    run_name: str = "",
    smoke_test: bool = False,
    persist_intermediates: bool = False,
) -> None:
    import json

    result = run_tubelet.remote(
        video_id=video_id,
        fps=fps,
        clip_start_global=clip_start_global,
        object_name=object_name,
        object_id=object_id,
        expected_frames=expected_frames,
        run_name=run_name,
        smoke_test=smoke_test,
        persist_intermediates=persist_intermediates,
    )
    print(json.dumps(result, indent=2))
