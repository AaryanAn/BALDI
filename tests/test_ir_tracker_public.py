import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking.ir_tracker import IRTracker


def test_ir_tracker_snapshot_paths_is_copy():
    t = IRTracker(camera_index=1)
    t.paths = [[(1, 2), (3, 4)]]
    snap = t.snapshot_paths()
    assert snap == [[(1, 2), (3, 4)]]
    snap[0][0] = (99, 99)
    assert t.paths[0][0] == (1, 2)


def test_ir_tracker_clear_path_resets():
    t = IRTracker(camera_index=1)
    t.paths = [[(1, 1)]]
    t.current_path = t.paths[0]
    t.drawing = True
    t.clear_path()
    assert t.paths == []
    assert t.current_path is None
    assert t.drawing is False
