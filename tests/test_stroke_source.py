import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_pages.stroke_source import clear_active_paths, snapshot_active_paths


class _G:
    def __init__(self):
        self.cleared = False
        self._paths = [[(1, 1)]]

    def snapshot_paths(self):
        return [list(p) for p in self._paths]

    def clear_path(self):
        self.cleared = True
        self._paths = []


class _I:
    def __init__(self):
        self.cleared = False
        self._paths = [[(2, 2)]]

    def snapshot_paths(self):
        return [list(p) for p in self._paths]

    def clear_path(self):
        self.cleared = True
        self._paths = []


def test_snapshot_gesture_when_ir_none():
    g, i = _G(), _I()
    assert snapshot_active_paths("gesture", g, None) == [[(1, 1)]]


def test_snapshot_gesture_when_ir_present_but_gesture_selected():
    g, i = _G(), _I()
    assert snapshot_active_paths("gesture", g, i) == [[(1, 1)]]


def test_snapshot_ir_when_selected():
    g, i = _G(), _I()
    assert snapshot_active_paths("ir", g, i) == [[(2, 2)]]


def test_clear_gesture_when_ir_none():
    g = _G()
    clear_active_paths("gesture", g, None)
    assert g.cleared is True


def test_clear_ir_when_selected():
    g, i = _G(), _I()
    clear_active_paths("ir", g, i)
    assert i.cleared is True and g.cleared is False


def test_clear_gesture_when_ir_present_but_gesture_selected():
    g, i = _G(), _I()
    clear_active_paths("gesture", g, i)
    assert g.cleared is True and i.cleared is False
