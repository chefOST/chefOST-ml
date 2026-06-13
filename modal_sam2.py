"""
chefOST onboarding — SAM2 video segmentation on Modal, logged to Comet.

Why this is the optimal shape:
  * SAM2 + checkpoints are baked into the image at BUILD time, so they are
    cached and never re-downloaded on a cold start.
  * The Comet API key is injected via a Modal Secret -> no interactive login,
    and every teammate runs it identically.
  * The input clip is passed from your laptop as bytes -> no manual upload.
  * Imports are deferred into the container, so `modal run` works even if you
    don't have torch/sam2 installed locally.

One-time setup (local machine):
    pip install modal
    modal token new
    modal secret create comet COMET_API_KEY=<your-comet-api-key>

Run:
    modal run modal_sam2.py --video cooking_clip.mp4
"""

import modal

app = modal.App("chefost-sam2")

# Only the API key is secret; workspace/project are safe to keep in code.
COMET_WORKSPACE = "chefost"
COMET_PROJECT = "sam2-onboarding"

# --- Build the image ONCE: clone SAM2, install it, bake in the checkpoints ---
sam2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "wget")
    .pip_install("torch", "torchvision", "numpy", "pillow", "comet_ml")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /sam2",
        "cd /sam2 && pip install -e .",
        "cd /sam2/checkpoints && ./download_ckpts.sh",  # SAM 2.1 checkpoints
    )
)

# These imports only run inside the container, not on your laptop.
with sam2_image.imports():
    import os
    import subprocess
    import comet_ml          # import before torch so auto-logging hooks attach
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

CKPT = "/sam2/checkpoints/sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _overlay(frame_rgb, mask, color=(0, 255, 0), alpha=0.5):
    out = frame_rgb.copy().astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * out[..., c][m] + alpha * color[c]
    return out.astype(np.uint8)


@app.function(
    image=sam2_image,
    gpu="L4",                                     # cheap Ada GPU, supports bf16
    timeout=900,
    secrets=[modal.Secret.from_name("comet")],    # injects COMET_API_KEY
)
def run_sam2(video_bytes: bytes, prompt_point=None, prompt_label=None, log_every: int = 10):
    os.makedirs("/tmp/frames", exist_ok=True)
    with open("/tmp/clip.mp4", "wb") as f:
        f.write(video_bytes)

    # SAM2's video predictor expects frames named 00000.jpg, 00001.jpg, ...
    subprocess.run(
        ["ffmpeg", "-i", "/tmp/clip.mp4", "-q:v", "2",
         "-start_number", "0", "/tmp/frames/%05d.jpg"],
        check=True,
    )
    frame_names = sorted(f for f in os.listdir("/tmp/frames") if f.endswith(".jpg"))

    # If no click point was given, default to the center of the first frame.
    # SAM2 will segment whatever object sits there (usually the food, which is
    # typically center-frame in cooking clips). Good enough for onboarding.
    if prompt_point is None:
        w, h = Image.open(f"/tmp/frames/{frame_names[0]}").size
        prompt_point = [[w // 2, h // 2]]
        prompt_label = [1]
    print(f"Using click point {prompt_point}")

    # COMET_API_KEY comes from the Modal Secret, so no login() call is needed.
    exp = comet_ml.start(
        workspace=COMET_WORKSPACE,
        project_name=COMET_PROJECT,
        experiment_config=comet_ml.ExperimentConfig(
            name="modal-sam2", tags=["sam2", "modal", "onboarding"]
        ),
    )
    exp.log_parameters({
        "model": "sam2.1_hiera_large",
        "prompt_point": prompt_point,
        "gpu": "L4",
    })

    predictor = build_sam2_video_predictor(CFG, CKPT)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(
            video_path="/tmp/frames",
            offload_video_to_cpu=True,    # keep frames in CPU RAM, not VRAM
            offload_state_to_cpu=True,    # keep the growing memory bank on CPU
        )
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.array(prompt_point, dtype=np.float32),
            labels=np.array(prompt_label, dtype=np.int32),
        )
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx % log_every != 0:
                continue
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            frame = np.array(
                Image.open(f"/tmp/frames/{frame_names[frame_idx]}").convert("RGB")
            )
            overlay = Image.fromarray(_overlay(frame, mask))
            exp.log_image(overlay, name=f"frame_{frame_idx:05d}")
            exp.log_metric("mask_area_px", int(mask.sum()), step=frame_idx)

    exp.end()
    print("Done — view the run at https://www.comet.com")


@app.local_entrypoint()
def main(video: str = "cooking_clip.mp4"):
    video_bytes = open(video, "rb").read()
    # No click point needed — the model defaults to the center of frame 0.
    # Once it works, you can pass your own point, e.g.:
    #   run_sam2.remote(video_bytes, [[450, 300]], [1])
    run_sam2.remote(video_bytes)