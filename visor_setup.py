"""
chefOST - download ONE VISOR clip (P01_107) and convert to DAVIS format.

This grabs just one clip's frames + masks (a few hundred MB, ~seconds),
NOT the 28GB archive. Enough to get your SAM2 run working today.
Everything lands in the 'visor-data' cloud volume.

RUN (from the VSCode terminal):
    modal run visor_one_clip.py
"""

import os

import modal

app = modal.App("chefost-visor-one-clip")

volume = modal.Volume.from_name("visor-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "git", "libgl1", "libglib2.0-0")
    .pip_install("numpy", "pillow", "opencv-python-headless",
                 "tqdm", "pandas", "scipy", "imgaug")
    .run_commands("git clone https://github.com/epic-kitchens/VISOR-VOS.git /VISOR-VOS")
)

BASE = "https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/GroundTruth-SparseAnnotations"
FRAMES_ZIP = f"{BASE}/rgb_frames/val/P01/P01_107.zip"
ANNOT_JSON = f"{BASE}/annotations/val/P01_107.json"


def download_if_missing(url, dest):
    """Download url -> dest only if dest doesn't already exist.

    Replaces `wget -nc`, which exits 1 (failure) when the file is already
    present, breaking `check=True` on re-runs. Doing the existence check in
    Python keeps setup idempotent without the bogus error.
    """
    import os, subprocess

    if os.path.exists(dest):
        print(f"Already have {dest}, skipping download.")
        return
    subprocess.run(f'wget -O "{dest}" "{url}"', shell=True, check=True)


@app.function(image=image, volumes={"/data": volume}, timeout=1800)
def setup():
    import subprocess

    # Build the folder layout the converter expects, inside the volume.
    os.makedirs("/data/sparse/annotations/val", exist_ok=True)
    os.makedirs("/data/sparse/rgb_frames/val", exist_ok=True)

    # 1) the annotation (masks) - one small json
    download_if_missing(ANNOT_JSON, "/data/sparse/annotations/val/P01_107.json")

    # 2) the frames - one zip, unzip into a per-video folder.
    # The converter reads frames from `{images_root}/{video}/{frame}.jpg`
    # (visor_to_davis.py joins images_root + datapoint["image"]["video"]),
    # so the frames MUST land under a P01_107/ subfolder, not flat in val/.
    download_if_missing(FRAMES_ZIP, "/tmp/P01_107.zip")
    # Clean up stray flat frames left in val/ by older runs (before the
    # per-video-subfolder fix). The converter ignores them, but they clutter
    # the persistent volume. The glob only matches *.jpg directly in val/, so
    # the P01_107/ subdir is left intact; `-f` makes it a no-op when none exist.
    subprocess.run("rm -f /data/sparse/rgb_frames/val/*.jpg", shell=True, check=True)
    subprocess.run("unzip -o /tmp/P01_107.zip -d /data/sparse/rgb_frames/val/P01_107/", shell=True, check=True)

    # Show what the frames unzipped into (so we confirm the layout).
    print("\n=== rgb_frames/val after unzip ===")
    subprocess.run("ls /data/sparse/rgb_frames/val", shell=True)
    print("\n=== one level deeper ===")
    subprocess.run("ls /data/sparse/rgb_frames/val/* | head", shell=True)

    # 3) convert to DAVIS format
    print("\n>>> converting...")
    subprocess.run(
        "python /VISOR-VOS/visor_to_davis.py "
        "-set val -keep_first_frame_masks_only 0 "
        "-visor_jsons_root /data/sparse/annotations "
        "-images_root /data/sparse/rgb_frames/val "
        "-output_directory /data/out_data",
        shell=True, check=True,
    )

    volume.commit()
    print("\nDONE -> /data/out_data/VISOR_2022")


@app.local_entrypoint()
def main():
    setup.remote()