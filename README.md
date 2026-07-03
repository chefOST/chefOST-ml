# ChefOST dataset generation

Builds a pseudo-labeled cooking video dataset for object state understanding.

## Single-video object state tracking

Track one selected cooking object with SAM2, classify sampled object crops with SmolVLM, smooth predictions, and emit a transition timeline plus visual evidence.

Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Local example:

```bash
python3 src/object_state_tracking.py \
  --video dataset/raw_videos/onion_001.mp4 \
  --object onion \
  --fps 1 \
  --out runs/onion_001 \
  --sam2-checkpoint /path/to/sam2_hiera_large.pt \
  --sam2-config sam2_hiera_l.yaml \
  --wandb-project chefost
```

If neither `--bbox` nor `--point` is provided, the script defaults to a lower-middle click on frame 0. If SAM2 is not installed, it falls back to a crude box around that point so the rest of the pipeline can be tested. Outputs are written to `runs/<id>/results.json`, `crops/`, `masks/`, and `overlays/`.

Modal example:

```bash
modal run modal_object_state_app.py --video dataset/raw_videos/onion_001.mp4 --obj onion --bbox 120,80,360,330 --fps 1
```

## Usage

1. Put raw videos under `dataset/raw_videos/`.
2. Add video metadata either in `configs/dataset_config.yaml` under `videos:` or in `dataset/metadata/video_metadata.json`:

```json
[
  {
    "video_id": "onion_001",
    "video_path": "raw_videos/onion_001.mp4",
    "title": "How to dice an onion",
    "object": "onion",
    "task": "dice onion"
  }
]
```

3. Create and activate an isolated Python environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Run the pipeline:

```bash
python src/run_pipeline.py --config configs/dataset_config.yaml
```

## Outputs

- `dataset/frame_dataset.jsonl`
- `dataset/temporal_dataset.jsonl`
- `splits/train.jsonl`, `splits/val.jsonl`, `splits/test.jsonl`
- `dataset/metadata/video_metadata.json`
- per-video frames, masks, crops, scores, pseudo-labels, features, and final metadata.

The detector/tracker/VLM/feature modules expose fallback implementations so the pipeline can run locally. Replace the internals of `detect_objects.py`, `track_masks.py`, `score_states.py`, and `extract_features.py` with Grounding DINO, SAM2, CLIP/SigLIP, and DINOv2 adapters when those models are available.
