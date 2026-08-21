from __future__ import annotations

import unittest

from modal_tubelet import _run_name, _safe_component


class ModalTubeletTests(unittest.TestCase):
    def test_safe_components_accept_expected_ids(self) -> None:
        self.assertEqual(
            _safe_component("S12_Sandwich_7150991-2470", "video_id"),
            "S12_Sandwich_7150991-2470",
        )

    def test_safe_components_reject_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            _safe_component("../overwrite", "run_name")

    def test_explicit_run_name_is_preserved(self) -> None:
        self.assertEqual(_run_name("trial-001"), "trial-001")


if __name__ == "__main__":
    unittest.main()
