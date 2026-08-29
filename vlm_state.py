"""
chefOST - VLM state-extraction stage.

Reads the ingredient crops written to the 'visor-data' volume by sam2_run.py
(Crops/<seq>/<frame>.png), asks LLaVA-NeXT for the object's structured state,
and logs one W&B run per sequence (group='vlm_P01_107', job_type='vlm_state')
with a per-frame table: frame_idx | frame_name | crop | object | form | color |
doneness | confidence | raw.

This is a SEPARATE stage that consumes the saved crops - it does NOT re-run SAM2.

RUN:
    modal run vlm_state.py
"""

import json
import re

import modal

app = modal.App("chefost-vlm-state")

volume = modal.Volume.from_name("visor-data")

# The exact keys we expect back from the VLM, in panel order.
STATE_KEYS = ("object", "category", "form", "color",
              "doneness", "texture", "description", "confidence")

PROMPT = (
    "You are a food-state annotator. The image is a close-up crop taken from an "
    "egocentric (head-mounted) kitchen video. It is centered on ONE object of "
    "interest, which may be partially occluded by a hand or the edge of the "
    "frame, and there may be background clutter you must ignore.\n"
    "\n"
    "Task: identify the single most prominent object at the CENTER of the crop "
    "and describe its current physical state. Base every field ONLY on what is "
    "actually visible. Never guess. If a field cannot be determined from the "
    "image, use the string \"unknown\".\n"
    "\n"
    "Field rules:\n"
    "- object: a specific noun for the centered object (e.g. \"tomato\", "
    "\"chopping knife\", \"white mug\"). Not a guess about what it might be.\n"
    "- category: one of \"food\", \"utensil\", \"container\", \"hand\", \"other\".\n"
    "- form: physical form. For food one of \"whole\", \"sliced\", \"diced\", "
    "\"chopped\", \"minced\", \"grated\", \"mashed\", \"liquid\", \"powder\", "
    "\"other\". For non-food use \"not_applicable\".\n"
    "- color: the dominant visible color(s) of the object.\n"
    "- doneness: the cooking state, ONLY if the object is food that could be "
    "cooked: one of \"raw\", \"partially_cooked\", \"cooked\", \"browned\", "
    "\"burnt\". If the object is not food, or is food not being cooked (e.g. a "
    "fresh vegetable), use \"not_applicable\".\n"
    "- texture: a visible surface cue, e.g. \"glossy\", \"dry\", \"wet\", "
    "\"crispy\", \"steaming\", \"charred\", \"smooth\", or \"unknown\".\n"
    "- description: ONE short sentence describing the object's current state.\n"
    "- confidence: your genuine confidence, a float 0-1, that the \"object\" "
    "identification is correct. Use a LOW value when the crop is blurry, "
    "occluded, or ambiguous.\n"
    "\n"
    "Respond with ONLY a single JSON object, no prose and no code fence, with "
    "exactly these keys: object, category, form, color, doneness, texture, "
    "description, confidence."
)


def parse_state(raw, keys=STATE_KEYS):
    """Extract a structured state dict from a VLM's raw text response.

    Pulls the first {...} JSON object out of `raw` (tolerating markdown fences
    and surrounding prose) and returns a dict with EXACTLY `keys`. Any key the
    model omitted, and every key if parsing fails, is set to None.
    """
    data = {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                data = parsed
        except (ValueError, TypeError):
            data = {}
    return {k: data.get(k) for k in keys}


MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", "transformers", "accelerate",
                 "pillow", "numpy", "wandb")
)


@app.function(
    image=image,
    gpu="a10g",                     # LLaVA-NeXT 7B won't fit comfortably on a T4
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=3600,
)
def run():
    import os, glob
    import torch
    from PIL import Image
    import wandb
    from transformers import (LlavaNextProcessor,
                              LlavaNextForConditionalGeneration)

    DATA = "/data/out_data/VISOR_2022"
    CROPS_ROOT = f"{DATA}/Crops"

    if not os.path.isdir(CROPS_ROOT):
        print(f"No crops at {CROPS_ROOT} - run sam2_run.py first.")
        return

    # --- load LLaVA-NeXT once, reuse across every crop ---
    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).to("cuda")

    def describe(pil_img):
        """Return the VLM's raw text answer for one crop."""
        conversation = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = processor(images=pil_img, text=prompt,
                           return_tensors="pt").to("cuda", torch.float16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = processor.decode(out[0], skip_special_tokens=True)
        # keep only the assistant turn (after the final instruction marker)
        return text.split("[/INST]")[-1].strip()

    # Only the food-object subsequences from the latest SAM2 run (bread,
    # cereal) - the volume still holds stale crop dirs from earlier
    # every-sequence runs, which we skip.
    FOOD_SEQS = {"P01_107_seq_00007", "P01_107_seq_00010"}
    sequences = sorted(s for s in os.listdir(CROPS_ROOT) if s in FOOD_SEQS)
    # LIMIT=1 (env var) processes just the first sequence - cheap smoke test
    limit = os.environ.get("LIMIT")
    if limit:
        sequences = sequences[:int(limit)]
    print(f"Found {len(sequences)} sequences with crops to process.")

    for seq in sequences:
        crop_files = sorted(glob.glob(f"{CROPS_ROOT}/{seq}/*.png"))
        if not crop_files:
            continue

        wandb.init(entity="chefOST", project="justin_runs",
                   name=f"vlm_state_{seq}", group="sam2_vlm_llava_next",
                   job_type="vlm_state", tags=["vlm", "llava-next", "food"],
                   config={"model": MODEL_ID, "seq": seq}, reinit=True)
        table = wandb.Table(columns=[
            "frame_idx", "frame_name", "crop",
            "object", "category", "form", "color",
            "doneness", "texture", "description", "confidence", "raw"])

        for f_idx, cpath in enumerate(crop_files):
            name = os.path.splitext(os.path.basename(cpath))[0]
            pil = Image.open(cpath).convert("RGB")
            raw = describe(pil)
            s = parse_state(raw)
            table.add_data(f_idx, name, wandb.Image(pil),
                           s["object"], s["category"], s["form"], s["color"],
                           s["doneness"], s["texture"], s["description"],
                           s["confidence"], raw)

        wandb.log({"panel": table})
        wandb.finish()
        print(f"[{seq}] read state for {len(crop_files)} crops.")

    print("DONE - open wandb.ai -> chefOST/justin_runs, group 'vlm_P01_107'.")


@app.local_entrypoint()
def main():
    run.remote()
