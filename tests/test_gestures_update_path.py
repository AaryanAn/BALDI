"""
Exercise Gestures.update_path without MediaPipe (uninitialized instances).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gestures.gestures import Gestures


def _bare_gestures():
    g = object.__new__(Gestures)
    g.prev_point = None
    g._was_pinching = False
    g.paths = []
    g.current_path = None
    g.drawing = False
    g.STROKE_SAMPLE_MIN_PX = 10
    return g


def test_pinch_start_appends_stroke_and_point():
    g = _bare_gestures()
    Gestures.update_path(g, (100, 100), True)
    assert g.drawing is True
    assert len(g.paths) == 1
    assert len(g.paths[0]) == 1
    assert g._was_pinching is True


def test_pinch_release_stops_drawing():
    g = _bare_gestures()
    g.prev_point = (100, 100)
    g._was_pinching = True
    g.current_path = []
    g.paths.append(g.current_path)
    g.drawing = True
    g.current_path.append((100, 100))

    Gestures.update_path(g, (120, 120), False)
    assert g.drawing is False
    assert g.current_path is None


def test_while_pinching_movement_appends_when_above_min_distance():
    g = _bare_gestures()
    Gestures.update_path(g, (100, 100), True)
    Gestures.update_path(g, (120, 120), True)
    assert len(g.paths[0]) >= 2
