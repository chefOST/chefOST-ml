from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chefost-object-state-tracking"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git", "wget")
    .pip_install(
        "opencv-python-headless",
        "Pillow",
        "PyYAML",
        "numpy",
        "torch",
        "torchvision",
        "accelerate",
        "transformers",
        "safetensors",
        "wandb",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /sam2",
        "cd /sam2 && pip install -e .",
        "cd /sam2/checkpoints && ./download_ckpts.sh",
    )
    .add_local_dir("src", "/root/src", copy=True)
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="L4", timeout=60 * 60, secrets=[modal.Secret.from_name("wandb")])
def track_states(
    video_bytes: bytes,
    video_name: str,
    obj: str,
    bbox: list[int] | None = None,
    point: list[int] | None = None,
    fps: float = 1.0,
    states: list[str] | None = None,
    sam2_checkpoint: str | None = "/sam2/checkpoints/sam2.1_hiera_large.pt",
    sam2_config: str | None = "configs/sam2.1/sam2.1_hiera_l.yaml",
    wandb_project: str | None = None,
) -> dict:
    from pathlib import Path
    import tempfile

    from src.object_state_tracking import run_pipeline

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        video_path = work / video_name
        video_path.write_bytes(video_bytes)
        return run_pipeline(
            video_path=str(video_path),
            obj=obj,
            out_dir=str(work / "run"),
            bbox=bbox,
            point=point,
            fps=fps,
            states=states,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            wandb_project=wandb_project,
        )


@app.local_entrypoint()
def main(
    video: str,
    obj: str = "onion",
    bbox: str | None = None,
    point: str | None = None,
    fps: float = 1.0,
    wandb_project: str | None = None,
):
    """Example: modal run modal_object_state_app.py --video onion.mp4 --obj onion --bbox 10,20,200,220"""
    b = [int(x) for x in bbox.split(",")] if bbox else None
    p = [int(x) for x in point.split(",")] if point else None
    data = Path(video).read_bytes()
    result = track_states.remote(data, Path(video).name, obj, bbox=b, point=p, fps=fps, wandb_project=wandb_project)
    print(result["timeline"])
