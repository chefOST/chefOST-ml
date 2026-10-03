"""Patch TubeletGraph so later-object tracking can be disabled at runtime.

Applied during the Modal image build. Inserts an early return into
``compute_save_all_tubes`` that is taken when the environment variable
``TUBELETGRAPH_SKIP_LATER_TRACKING=1`` is set. The loop it skips re-tracks
newly found entities every ``collect_spacing`` frames to the end of the
video, which scales roughly quadratically with frame count.
"""

from __future__ import annotations

import sys
from pathlib import Path

ENV_FLAG = "TUBELETGRAPH_SKIP_LATER_TRACKING"
MARKER = "    print('Running later object tracking...')\n"
GUARD = (
    f"    if os.environ.get('{ENV_FLAG}') == '1':\n"
    f"        print('Skipping later object tracking ({ENV_FLAG}=1)')\n"
    "        save_all_to_json({'all_tracks': all_tracks, 'tracked_objs': tracked_objs}, final_all_tracks_path, dim=dim)\n"
    "        return\n"
)


def patch(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if GUARD in source:
        return False
    if source.count(MARKER) != 1:
        raise SystemExit(f"expected exactly one later-tracking marker in {path}")
    path.write_text(source.replace(MARKER, GUARD + MARKER), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    repository = Path(argv[1] if len(argv) > 1 else "/opt/TubeletGraph")
    target = repository / "TubeletGraph" / "tubelet" / "compute_tubelets_sam.py"
    changed = patch(target)
    print(f"{'patched' if changed else 'already patched'}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
