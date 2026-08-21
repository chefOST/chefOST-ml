#!/usr/bin/env python3
"""Create a one-channel TubeletGraph mask from polygons or an interactive UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


WINDOW = "Mask: click | n next polygon | ENTER save | u undo | r reset | q quit"


def load_polygon_file(path: Path, image_shape: tuple[int, int]) -> tuple[int, list]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    height, width = image_shape
    if int(payload["image_width"]) != width or int(payload["image_height"]) != height:
        raise ValueError(
            "Polygon dimensions do not match the input image: "
            f"polygon={payload['image_width']}x{payload['image_height']}, "
            f"image={width}x{height}"
        )
    polygons = [np.asarray(points, dtype=np.int32) for points in payload["polygons"]]
    if not polygons or any(len(points) < 3 for points in polygons):
        raise ValueError("Every polygon must contain at least three points")
    return int(payload.get("object_id", 1)), polygons


def save_mask(
    out_path: Path,
    image_shape: tuple[int, int],
    polygons: list[np.ndarray],
    object_id: int,
) -> None:
    if not 1 <= object_id <= 254:
        raise ValueError("object_id must be between 1 and 254")
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, polygons, object_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # TubeletGraph expects indexed PNG annotations and treats pixel indices as
    # object IDs, so preserve those indices in palette (P) mode.
    indexed_mask = Image.frombytes(
        "P", (mask.shape[1], mask.shape[0]), mask.tobytes()
    )
    indexed_mask.putpalette(
        [channel for index in range(256) for channel in (index, index, index)]
    )
    indexed_mask.save(out_path, "PNG")
    print(f"Saved {out_path}")
    print(f"Shape: {mask.shape}; values: {np.unique(mask).tolist()}")
    print(f"Object pixels: {int(np.count_nonzero(mask == object_id))}")


def interactive_polygons(image: np.ndarray) -> list[np.ndarray]:
    completed: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def redraw() -> None:
        preview = image.copy()
        overlay = image.copy()
        arrays = [np.asarray(points, dtype=np.int32) for points in completed]
        if arrays:
            cv2.fillPoly(overlay, arrays, (0, 255, 0))
            preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)
            cv2.polylines(preview, arrays, True, (0, 255, 0), 2)
        if current:
            points = np.asarray(current, dtype=np.int32)
            cv2.polylines(preview, [points], False, (0, 255, 255), 2)
            for point in current:
                cv2.circle(preview, point, 4, (0, 0, 255), -1)
        cv2.imshow(WINDOW, preview)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n"):
            if len(current) < 3:
                print("Need at least three points before closing this polygon.")
                continue
            completed.append(current.copy())
            current.clear()
            redraw()
        elif key in (8, 127, ord("u")):
            if current:
                current.pop()
            elif completed:
                current.extend(completed.pop())
            redraw()
        elif key == ord("r"):
            completed.clear()
            current.clear()
            redraw()
        elif key in (10, 13):
            if current:
                if len(current) < 3:
                    print("Need at least three points in the current polygon.")
                    continue
                completed.append(current.copy())
                current.clear()
            if not completed:
                print("Draw at least one polygon.")
                continue
            break
        elif key == ord("q"):
            completed.clear()
            break

    cv2.destroyAllWindows()
    return [np.asarray(points, dtype=np.int32) for points in completed]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--polygons-json", type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    image_shape = image.shape[:2]

    if args.polygons_json:
        object_id, polygons = load_polygon_file(args.polygons_json, image_shape)
    else:
        object_id = args.object_id
        polygons = interactive_polygons(image)
        if not polygons:
            print("No mask saved.")
            return

    save_mask(args.out, image_shape, polygons, object_id)


if __name__ == "__main__":
    main()
