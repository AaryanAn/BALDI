import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_pages.frame_routing import raw_frame_for_active_source


def test_raw_frame_follows_gesture_source():
    g = np.zeros((2, 2, 3), dtype=np.uint8)
    g[0, 0] = (1, 2, 3)
    ir = np.zeros((2, 2, 3), dtype=np.uint8)
    ir[1, 1] = (9, 9, 9)
    out = raw_frame_for_active_source("gesture", g, ir)
    assert out is g


def test_raw_frame_follows_ir_source():
    g = np.zeros((2, 2, 3), dtype=np.uint8)
    ir = np.zeros((2, 2, 3), dtype=np.uint8)
    ir[1, 1] = (9, 9, 9)
    out = raw_frame_for_active_source("ir", g, ir)
    assert out is ir
