import numpy as np

from crop import crop_mask


def _mask(h, w, r0, r1, c0, c1):
    """Boolean mask of shape (h, w), True on rows r0..r1, cols c0..c1 inclusive."""
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1 + 1, c0:c1 + 1] = True
    return m


def test_centered_mask_gives_padded_bbox_size():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = _mask(100, 100, 40, 59, 40, 59)  # 20x20 box -> pad 25% = 5 each side
    crop = crop_mask(img, mask)
    assert crop.shape == (30, 30, 3)


def test_edge_hugging_mask_clamps_to_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = _mask(50, 50, 0, 19, 0, 19)  # 20x20 box at top-left, pad 5
    crop = crop_mask(img, mask)
    # low side clamps to 0, high side extends to 24 -> 25 pixels
    assert crop.shape == (25, 25, 3)


def test_empty_mask_returns_none():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)
    assert crop_mask(img, mask) is None


def test_default_keeps_background_pixels():
    img = np.full((100, 100, 3), 10, dtype=np.uint8)  # non-white background
    mask = _mask(100, 100, 40, 59, 40, 59)
    crop = crop_mask(img, mask)  # whiteout=False by default
    assert (crop[0, 0] == 10).all()  # padding corner unchanged


def test_whiteout_blanks_outside_mask_only():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # black everywhere
    mask = _mask(100, 100, 40, 59, 40, 59)
    crop = crop_mask(img, mask, whiteout=True)
    assert (crop[0, 0] == 255).all()      # padding corner -> white
    assert (crop[14, 14] == 0).all()      # center (inside mask) -> unchanged black
