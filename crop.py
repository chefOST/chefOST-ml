"""Crop the masked ingredient region out of a frame for VLM input."""

import numpy as np


def crop_mask(img, mask, pad_frac=0.25, whiteout=False):
    """Crop the padded bounding box of `mask` out of `img`.

    img:  H×W×3 uint8 RGB frame
    mask: H×W bool / {0,1} predicted mask
    pad_frac: fraction of the bbox width/height added on each side (default 0.25)
    whiteout: if True, paint pixels outside the mask white before cropping
    returns: cropped H'×W'×3 uint8 array, or None if the mask is empty
    """
    mask = np.asarray(mask).astype(bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = mask.shape
    pad_r = int(round(pad_frac * (rmax - rmin + 1)))
    pad_c = int(round(pad_frac * (cmax - cmin + 1)))

    r0 = max(0, rmin - pad_r)
    r1 = min(h, rmax + 1 + pad_r)
    c0 = max(0, cmin - pad_c)
    c1 = min(w, cmax + 1 + pad_c)

    out = img[r0:r1, c0:c1].copy()
    if whiteout:
        out[~mask[r0:r1, c0:c1]] = 255
    return out
