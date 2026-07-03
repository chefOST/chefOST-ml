"""
chefOST - run SAM2 over every VISOR subsequence and log each to Weights & Biases.

What it does:
  * Reads the DAVIS-format clips from the 'visor-data' volume.
  * For each P01_107_seq_* subsequence: prompts SAM2 with the frame-0 ground-truth
    mask of its first object, then propagates through the clip.
  * Logs each frame + predicted-mask overlay to W&B as ONE run per sequence
    (name = m1_sam2_<seq>).
  * Scores predictions vs ground truth (IoU) on every annotated later frame, logs
    per-sequence mean IoU, and prints an overall summary across sequences.

One-time setup before running:
    modal secret create wandb WANDB_API_KEY=<your key from wandb.ai/authorize>

RUN:
    modal run sam2_run.py
"""

import modal

app = modal.App("chefost-sam2-run")

volume = modal.Volume.from_name("visor-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch", "torchvision", "numpy", "pillow",
                 "opencv-python-headless", "wandb", "tqdm")
    .run_commands(
        "git clone https://github.com/facebookresearch/sam2.git /sam2",
        "cd /sam2 && pip install -e .",
        "cd /sam2/checkpoints && ./download_ckpts.sh",
    )
    # ship crop.py to the container so `from crop import crop_mask` resolves
    .add_local_python_source("crop")
)


@app.function(
    image=image,
    gpu="t4",                       # cheapest GPU; plenty for these short clips
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=3600,
)
def run():
    import os, glob, shutil
    import numpy as np
    import torch
    from PIL import Image
    import wandb
    from sam2.build_sam import build_sam2_video_predictor
    from crop import crop_mask

    DATA = "/data/out_data/VISOR_2022"
    JPEG_ROOT = f"{DATA}/JPEGImages/480p"
    ANNO_ROOT = f"{DATA}/Annotations/480p"

    # Build SAM2 ONCE and reuse the predictor across every sequence.
    ckpt = "/sam2/checkpoints/sam2.1_hiera_large.pt"
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(cfg, ckpt)

    def eval_one(seq):
        """Run + score one sequence into the currently-active W&B run.

        Returns the sequence mean IoU, or None if it has nothing to score.
        """
        frames_dir = f"{JPEG_ROOT}/{seq}"
        annot_dir = f"{ANNO_ROOT}/{seq}"

        # --- frame-0 ground-truth mask -> SAM2 prompt ---
        annot_files = sorted(glob.glob(f"{annot_dir}/*.png"))
        if not annot_files:
            print(f"[{seq}] no annotations, skipping.")
            return None
        gt0 = np.array(Image.open(annot_files[0]))
        object_values = [int(v) for v in np.unique(gt0) if v != 0]
        if not object_values:
            print(f"[{seq}] frame 0 has no objects, skipping.")
            return None
        target = object_values[0]      # track the first object in frame 0
        print(f"[{seq}] objects in frame 0: {object_values} -> tracking value {target}")
        prompt_mask = (gt0 == target)

        # all available ground-truth masks for this object, keyed by frame name
        gt_by_name = {}
        for p in annot_files:
            name = os.path.splitext(os.path.basename(p))[0]
            gt_by_name[name] = (np.array(Image.open(p)) == target)

        frame_files = sorted(glob.glob(f"{frames_dir}/*.jpg"))
        frame_names = [os.path.splitext(os.path.basename(p))[0] for p in frame_files]

        # SAM2's init_state sorts a frame folder by int(filename), assuming names like
        # 00000.jpg. VISOR frames are 'P01_107_frame_0000000155.jpg', which int() can't
        # parse. Stage a per-sequence dir of symlinks renamed 00000.jpg..NNNNN.jpg in the
        # SAME sorted order, so SAM2's f_idx lines up with frame_files / frame_names.
        staging_dir = f"/tmp/sam2_frames/{seq}"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir)
        for i, src in enumerate(frame_files):
            os.symlink(os.path.abspath(src), f"{staging_dir}/{i:05d}.jpg")

        table = wandb.Table(columns=["frame_idx", "result", "iou", "crop"])
        ious = []

        crops_dir = f"{DATA}/Crops/{seq}"
        os.makedirs(crops_dir, exist_ok=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=staging_dir)
            predictor.add_new_mask(state, frame_idx=0, obj_id=target, mask=prompt_mask)

            for f_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
                pred = (mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8)

                img = np.array(Image.open(frame_files[f_idx]).convert("RGB"))
                # safety: match mask size to image if SAM returned a different resolution
                if pred.shape != img.shape[:2]:
                    pred = np.array(Image.fromarray(pred).resize(
                        (img.shape[1], img.shape[0]), Image.NEAREST))

                wb_img = wandb.Image(img, masks={
                    "prediction": {"mask_data": pred, "class_labels": {1: f"obj_{target}"}}
                })

                iou = None
                name = frame_names[f_idx]
                if name in gt_by_name and f_idx != 0:   # don't score the prompt frame
                    g = gt_by_name[name].astype(np.uint8)
                    inter = np.logical_and(pred, g).sum()
                    union = np.logical_or(pred, g).sum()
                    if union > 0:
                        iou = float(inter / union)
                        ious.append(iou)

                # crop the masked ingredient region for downstream VLM input
                crop = crop_mask(img, pred)   # 25% padding, no white-out (see crop.py)
                wb_crop = None
                if crop is not None:
                    Image.fromarray(crop).save(f"{crops_dir}/{name}.png")
                    wb_crop = wandb.Image(crop)

                table.add_data(f_idx, wb_img, iou, wb_crop)

        wandb.log({"panel": table})
        mean_iou = sum(ious) / len(ious) if ious else None
        if mean_iou is not None:
            wandb.log({"mean_iou": mean_iou, "scored_frames": len(ious)})
            print(f"[{seq}] mean IoU: {mean_iou:.3f} over {len(ious)} frames")
        else:
            wandb.log({"scored_frames": 0})
            print(f"[{seq}] no scoreable later frames (object left view / no overlap).")
        return mean_iou

    # --- loop over every subsequence; one W&B run each ---
    sequences = sorted(os.listdir(JPEG_ROOT))
    print(f"Found {len(sequences)} sequences to evaluate.")

    results = {}
    for seq in sequences:
        wandb.init(entity="chefOST", project="justin_runs", name=f"m1_sam2_{seq}",
                   group="sam2_vlm_llava_next", tags=["sam2", "sam2.1_hiera_large"],
                   config={"tracker": "sam2.1_hiera_large", "seq": seq}, reinit=True)
        try:
            results[seq] = eval_one(seq)
        except Exception as e:
            print(f"[{seq}] FAILED: {e}")
            results[seq] = None
        finally:
            wandb.finish()

    # persist the crops written to the volume this run
    volume.commit()

    # --- overall summary across sequences ---
    print("\n==== SUMMARY ====")
    for seq in sequences:
        v = results[seq]
        print(f"  {seq}: {('%.3f' % v) if v is not None else '—'}")
    scored = [v for v in results.values() if v is not None]
    if scored:
        print(f"\nOverall mean IoU across {len(scored)}/{len(sequences)} scored "
              f"sequences: {sum(scored) / len(scored):.3f}")
    print("DONE - open wandb.ai -> chefOST/justin_runs to see all runs.")


@app.local_entrypoint()
def main():
    run.remote()
