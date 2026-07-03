from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Allow `python src/object_state_tracking.py` from repository root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from src.utils import ensure_dir, write_json, load_yaml

DEFAULT_STATES = ["whole", "halved", "sliced", "chopped", "diced", "minced", "unknown"]
DEFAULT_TRANSITIONS = {
    "whole": {"whole", "halved", "sliced", "chopped", "unknown"},
    "halved": {"halved", "sliced", "chopped", "diced", "unknown"},
    "sliced": {"sliced", "chopped", "diced", "minced", "unknown"},
    "chopped": {"chopped", "diced", "minced", "unknown"},
    "diced": {"diced", "minced", "unknown"},
    "minced": {"minced", "unknown"},
    "unknown": set(DEFAULT_STATES),
}


def sample_frames(video_path: str | Path, out_dir: str | Path, fps: float = 1.0) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from e

    out = ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(source_fps / fps)), 1)
    rows: list[dict[str, Any]] = []
    frame_idx = saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            timestamp = frame_idx / source_fps
            frame_id = f"{saved_idx:06d}"
            path = out / f"{frame_id}.jpg"
            cv2.imwrite(str(path), frame)
            rows.append({"frame_index": frame_idx, "frame_id": frame_id, "timestamp": timestamp, "image_path": str(path)})
            saved_idx += 1
        frame_idx += 1
    cap.release()
    return rows


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    b = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    if b[0] >= b[2] or b[1] >= b[3]:
        raise ValueError(f"bbox {bbox} is outside the frame {width}x{height}")
    return b


def _mask_from_bbox(image_path: str | Path, bbox: list[int]) -> np.ndarray:
    im = Image.open(image_path)
    mask = np.zeros((im.height, im.width), dtype=np.uint8)
    x1, y1, x2, y2 = _clamp_bbox(bbox, im.width, im.height)
    mask[y1:y2, x1:x2] = 255
    return mask


def track_with_sam2(frames: list[dict[str, Any]], out_dir: str | Path, bbox: list[int] | None = None,
                    point: list[int] | None = None, checkpoint: str | None = None,
                    model_cfg: str | None = None) -> list[dict[str, Any]]:
    """Track target with SAM2 when installed; otherwise use a static bbox fallback."""
    out = ensure_dir(out_dir)
    rows: list[dict[str, Any]] = []
    try:
        import torch  # type: ignore
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
    except Exception:
        if bbox is None:
            if point is None:
                raise RuntimeError("SAM2 is unavailable and fallback tracking requires --bbox or --point")
            im = Image.open(frames[0]["image_path"])
            x, y = point
            # ponytail: crude local fallback only; use SAM2/GroundingDINO for real masks.
            bbox = [max(0, x - 80), max(0, y - 80), min(im.width, x + 80), min(im.height, y + 80)]
        for f in frames:
            im = Image.open(f["image_path"])
            bbox = _clamp_bbox(bbox, im.width, im.height)
            mask = _mask_from_bbox(f["image_path"], bbox)
            mask_path = out / f"{f['frame_id']}.png"
            Image.fromarray(mask).save(mask_path)
            rows.append({**f, "mask_path": str(mask_path), "bbox": bbox, "mask_area": int((mask > 0).sum()), "tracker": "static_bbox_fallback"})
        return rows

    if checkpoint is None or model_cfg is None:
        raise ValueError("SAM2 tracking requires checkpoint and model_cfg")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    frame_dir = Path(frames[0]["image_path"]).parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        state = predictor.init_state(video_path=str(frame_dir))
        if bbox is not None:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=np.array(bbox, dtype=np.float32))
        elif point is not None:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, points=np.array([point], dtype=np.float32), labels=np.array([1], dtype=np.int32))
        else:
            raise ValueError("Provide either bbox or point")
        by_idx = {i: f for i, f in enumerate(frames)}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            if 1 not in obj_ids:
                continue
            j = list(obj_ids).index(1)
            mask = (mask_logits[j].detach().cpu().numpy() > 0).squeeze().astype(np.uint8) * 255
            ys, xs = np.where(mask > 0)
            if len(xs):
                mb = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            else:
                mb = [0, 0, 0, 0]
            f = by_idx[frame_idx]
            mask_path = out / f"{f['frame_id']}.png"
            Image.fromarray(mask).save(mask_path)
            rows.append({**f, "mask_path": str(mask_path), "bbox": mb, "mask_area": int((mask > 0).sum()), "tracker": "sam2"})
    return sorted(rows, key=lambda r: r["timestamp"])


def create_crop(image_path: str | Path, mask_path: str | Path, bbox: list[int], out_path: str | Path) -> str:
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
    masked = Image.composite(image, bg, mask)
    crop = masked.crop(tuple(_clamp_bbox(bbox, image.width, image.height))).convert("RGB")
    ensure_dir(Path(out_path).parent)
    crop.save(out_path)
    return str(out_path)


class HeuristicStateClassifier:
    """Lightweight local fallback; useful for testing the pipeline without a VLM."""

    def __init__(self, states: list[str]):
        self.states = states

    def classify(self, image_path: str | Path, obj: str) -> tuple[str, str]:
        im = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.asarray(im).astype(np.float32) / 255.0
        non_black = arr[arr.mean(axis=2) > 0.03]
        if non_black.size == 0:
            non_black = arr.reshape(-1, 3)
        gray = arr.mean(axis=2)
        texture = float(np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1))))
        order = {s: i / max(len(self.states) - 1, 1) for i, s in enumerate(self.states)}
        label = min(self.states, key=lambda s: abs(texture * 6.0 - order[s]))
        if texture < 0.025 and "whole" in self.states:
            label = "whole"
        raw = f"heuristic texture={texture:.4f} label={label}"
        return label, raw


class SmolVLMStateClassifier:
    def __init__(self, states: list[str], model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        self.states = states
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            try:
                # transformers 4.x
                from transformers import AutoModelForVision2Seq as AutoVLM  # type: ignore
            except ImportError:
                # transformers 5.x / newer SmolVLM examples
                from transformers import AutoModelForImageTextToText as AutoVLM  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install compatible transformers, torch, and accelerate to use SmolVLM") from e
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoVLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def classify(self, image_path: str | Path, obj: str) -> tuple[str, str]:
        prompt = f"Classify the {obj} state as exactly one of: {', '.join(self.states)}. Answer with only the label."
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=12, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip().lower()
        if not raw:
            raw = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
        words = raw.replace(".", " ").replace(",", " ").replace(":", " ").split()
        label = next((s for s in self.states if s in words), "unknown")
        return label, raw


def smooth_labels(labels: list[str], states: list[str], window: int = 3, enforce_transitions: bool = True) -> list[str]:
    half = window // 2
    smoothed: list[str] = []
    for i in range(len(labels)):
        vote = Counter(labels[max(0, i - half):min(len(labels), i + half + 1)]).most_common(1)[0][0]
        if enforce_transitions and smoothed:
            allowed = DEFAULT_TRANSITIONS.get(smoothed[-1], set(states))
            if vote not in allowed:
                vote = smoothed[-1] if labels[i] not in allowed else labels[i]
        smoothed.append(vote)
    return smoothed


def transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev = None
    for r in rows:
        label = r["label"]
        if label != prev:
            out.append({"timestamp": r["timestamp"], "label": label, "frame_id": r["frame_id"], "crop_path": r.get("crop_path")})
            prev = label
    return out


def write_overlay(image_path: str | Path, mask_path: str | Path, label: str, out_path: str | Path) -> str:
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    red = Image.new("RGBA", image.size, (255, 0, 0, 90))
    image = Image.composite(red, image, mask).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), label, fill=(255, 255, 255))
    ensure_dir(Path(out_path).parent)
    image.save(out_path)
    return str(out_path)


def run_pipeline(video_path: str, obj: str, out_dir: str, bbox: list[int] | None = None, point: list[int] | None = None,
                 fps: float = 1.0, states: list[str] | None = None, sam2_checkpoint: str | None = None,
                 sam2_config: str | None = None, vlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
                 classifier_name: str = "smolvlm", wandb_project: str | None = None) -> dict[str, Any]:
    states = states or DEFAULT_STATES
    out = ensure_dir(out_dir)
    frames = sample_frames(video_path, out / "frames", fps=fps)
    if bbox is None and point is None:
        im = Image.open(frames[0]["image_path"])
        point = [im.width // 2, int(im.height * 0.78)]
    masks = track_with_sam2(frames, out / "masks", bbox=bbox, point=point, checkpoint=sam2_checkpoint, model_cfg=sam2_config)
    classifier = HeuristicStateClassifier(states) if classifier_name == "heuristic" else SmolVLMStateClassifier(states, vlm_model)

    rows: list[dict[str, Any]] = []
    for m in masks:
        crop_path = out / "crops" / f"{m['frame_id']}.png"
        create_crop(m["image_path"], m["mask_path"], m["bbox"], crop_path)
        label, raw = classifier.classify(crop_path, obj)
        rows.append({**m, "crop_path": str(crop_path), "raw_label": label, "raw_response": raw})
    labels = smooth_labels([r["raw_label"] for r in rows], states)
    for r, label in zip(rows, labels):
        r["label"] = label
        r["overlay_path"] = write_overlay(r["image_path"], r["mask_path"], label, out / "overlays" / f"{r['frame_id']}.jpg")
        Path(r["mask_path"]).unlink(missing_ok=True)
        r.pop("mask_path", None)
    result = {"object": obj, "states": states, "frames": rows, "timeline": transitions(rows)}
    write_json(out / "results.json", result)

    if wandb_project:
        import wandb  # type: ignore
        run = wandb.init(project=wandb_project, config={"video_path": video_path, "object": obj, "fps": fps, "states": states})
        table = wandb.Table(columns=["timestamp", "frame_id", "overlay", "crop", "raw_label", "label", "raw_response", "mask_area"])
        for r in rows:
            table.add_data(r["timestamp"], r["frame_id"], wandb.Image(r["overlay_path"]), wandb.Image(r["crop_path"]), r["raw_label"], r["label"], r["raw_response"], r["mask_area"])
        wandb.log({"state_tracking": table, "transitions": result["timeline"]})
        run.finish()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Track one cooking object and classify state transitions")
    p.add_argument("--video", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--out", default="runs/object_state")
    p.add_argument("--bbox", nargs=4, type=int, help="x1 y1 x2 y2 prompt on frame 0")
    p.add_argument("--point", nargs=2, type=int, help="x y positive click on frame 0; defaults to lower-middle")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    p.add_argument("--sam2-checkpoint")
    p.add_argument("--sam2-config")
    p.add_argument("--vlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--classifier", choices=["smolvlm", "heuristic"], default="smolvlm")
    p.add_argument("--wandb-project")
    p.add_argument("--config")
    args = p.parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    result = run_pipeline(
        video_path=args.video,
        obj=args.object,
        out_dir=args.out,
        bbox=args.bbox or cfg.get("bbox"),
        point=args.point or cfg.get("point"),
        fps=args.fps or cfg.get("fps", 1.0),
        states=args.states or cfg.get("states", DEFAULT_STATES),
        sam2_checkpoint=args.sam2_checkpoint or cfg.get("sam2_checkpoint"),
        sam2_config=args.sam2_config or cfg.get("sam2_config"),
        vlm_model=args.vlm_model or cfg.get("vlm_model", "HuggingFaceTB/SmolVLM-500M-Instruct"),
        classifier_name=args.classifier or cfg.get("classifier", "smolvlm"),
        wandb_project=args.wandb_project or cfg.get("wandb_project"),
    )
    print(json.dumps(result["timeline"], indent=2))


if __name__ == "__main__":
    main()
