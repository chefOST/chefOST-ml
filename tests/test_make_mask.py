from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.make_mask import save_mask


class MakeMaskTests(unittest.TestCase):
    def test_save_mask_writes_indexed_png_and_preserves_object_id(self) -> None:
        polygon = np.asarray([(1, 1), (6, 1), (6, 6), (1, 6)], dtype=np.int32)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mask.png"
            save_mask(output, (8, 8), [polygon], object_id=7)

            with Image.open(output) as mask:
                self.assertEqual(mask.mode, "P")
                self.assertEqual(sorted(np.unique(np.asarray(mask)).tolist()), [0, 7])


if __name__ == "__main__":
    unittest.main()
