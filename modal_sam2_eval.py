"""
chefOST — validate SAM2 on the DAVIS 2017 benchmark, logged to Comet.

What it does:
  * Loads DAVIS 2017 (frames + per-frame ground-truth masks) baked into the image.
  * For each sequence, prompts SAM2 with the GROUND-TRUTH mask of frame 0
    (the standard "semi-supervised VOS" protocol), then propagates.
  * Scores every predicted frame against the true mask using the OFFICIAL
    DAVIS metrics: J (region IoU), F (boundary accuracy), and J&F (their mean).
  * Logs per-sequence and overall metrics to Comet -> multiple evaluations,
    multiple metrics.

One-time setup (same as before):
    modal secret create comet COMET_API_KEY=<key>     # if not already done

Run:
    modal run modal_sam2_eval.py --max-sequences 3
"""

import modal

app = modal.App("chefost-sam2-eval")

COMET_WORKSPACE = "chefost"
COMET_PROJECT = "sam2-davis-eval"

# Official 480p train/val archive. If this 404s, get the current link from
# https://davischallenge.org/davis2017/code.html
DAVIS_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "numpy", "pillow", "comet_ml")
    .run_commands(
        # SAM2 + checkpoints
        "git clone https://github.com/facebookresearch/sam2.git /sam2",
        "cd /sam2 && pip install -e .",
        "cd /sam2/checkpoints && ./download_ckpts.sh",
        # Official DAVIS J&F metric functions
        "git clone https://github.com/davisvideochallenge/davis2017-evaluation.git /davis-eval",
        "cd /davis-eval && pip install -e .",
        # The dataset itself (frames + ground-truth masks)
        f"wget -q {DAVIS_URL} -O /tmp/davis.zip",
        "unzip -q /tmp/davis.zip -d /data && rm /tmp/davis.zip",
    )
)

with eval_image.imports():
    import os
    import comet_ml
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    from davis2017.metrics import db_eval_iou, db_eval_boundary  # official J and F

# Short name -> (checkpoint file, config) for each SAM 2.1 size.
MODELS = {
    "tiny":  ("sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt",     "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base":  ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt",     "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
DAVIS_ROOT = "/data/DAVIS"
JPEG = f"{DAVIS_ROOT}/JPEGImages/480p"
ANNO = f"{DAVIS_ROOT}/Annotations/480p"


@app.function(
    image=eval_image,
    gpu="L4",
    timeout=3600,
    secrets=[modal.Secret.from_name("comet")],
)
def evaluate(max_sequences: int = 3, model: str = "large"):
    ckpt_name, cfg = MODELS[model]
    exp = comet_ml.start(
        workspace=COMET_WORKSPACE,
        project_name=COMET_PROJECT,
        experiment_config=comet_ml.ExperimentConfig(
            name=f"sam2-{model}", tags=["sam2", "davis2017", "validation", model]
        ),
    )
    exp.log_parameters({
        "model": ckpt_name,
        "model_size": model,
        "dataset": "DAVIS2017-val",
        "max_sequences": max_sequences,
    })

    predictor = build_sam2_video_predictor(cfg, f"/sam2/checkpoints/{ckpt_name}")
    sequences = sorted(os.listdir(JPEG))[:max_sequences]

    all_j, all_f = [], []
    for seq in sequences:
        frame_dir, ann_dir = f"{JPEG}/{seq}", f"{ANNO}/{seq}"
        frame_names = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))

        # Frame-0 ground truth: pixel value = object id (0 = background, 255 = void)
        ann0 = np.array(Image.open(f"{ann_dir}/00000.png"))
        obj_ids = [int(i) for i in np.unique(ann0) if i not in (0, 255)]
        if not obj_ids:
            continue

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=frame_dir)
            # Prompt each object with its TRUE frame-0 mask.
            for oid in obj_ids:
                predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=oid,
                    mask=(ann0 == oid),
                )
            # Propagate and collect predicted masks for every frame/object.
            preds = {}
            for f_idx, out_ids, logits in predictor.propagate_in_video(state):
                preds[f_idx] = {
                    int(oid): (logits[k] > 0.0).cpu().numpy().squeeze()
                    for k, oid in enumerate(out_ids)
                }

        # Score predictions vs ground truth, per object, per frame.
        seq_j, seq_f = [], []
        for f_idx, fname in enumerate(frame_names):
            ann_path = f"{ann_dir}/{fname.replace('.jpg', '.png')}"
            if not os.path.exists(ann_path):
                continue
            gt = np.array(Image.open(ann_path))
            for oid in obj_ids:
                gt_m = (gt == oid)
                pred_m = preds.get(f_idx, {}).get(oid, np.zeros_like(gt_m))
                seq_j.append(float(db_eval_iou(gt_m, pred_m)))
                seq_f.append(float(db_eval_boundary(gt_m, pred_m)))

        j, f = float(np.mean(seq_j)), float(np.mean(seq_f))
        exp.log_metrics({"J": j, "F": f, "J&F": (j + f) / 2}, prefix=seq)
        all_j += seq_j
        all_f += seq_f
        print(f"{seq:25s}  J={j:.3f}  F={f:.3f}  J&F={(j + f) / 2:.3f}")

    # Overall numbers across all evaluated sequences.
    J, F = float(np.mean(all_j)), float(np.mean(all_f))
    exp.log_metrics({"J": J, "F": F, "J&F": (J + F) / 2}, prefix="overall")
    print(f"\n{'OVERALL':25s}  J={J:.3f}  F={F:.3f}  J&F={(J + F) / 2:.3f}")
    print("Reference: SAM2 paper reports ~0.907 J&F on full DAVIS-2017 val.")

    exp.end()


@app.local_entrypoint()
def main(max_sequences: int = 3, model: str = "large"):
    evaluate.remote(max_sequences, model)