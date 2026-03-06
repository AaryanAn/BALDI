import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from evaluation.dtw import dtw_distance
from evaluation.letters import LetterEvaluator
from trajectory.normalization import preprocess_paths


def test_dtw_zero_for_identical():
    a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    d = dtw_distance(a, b)
    assert d == 0.0


def test_dtw_larger_for_shifted():
    a = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.5, 0.0], [1.5, 0.0]], dtype=np.float32)
    d1 = dtw_distance(a, b)
    d2 = dtw_distance(a, a)
    assert d1 > d2


def test_letter_evaluator_simple_flow(tmp_path):
    evalr = LetterEvaluator(tmp_path)

    paths = [[(0.0, 0.0), (1.0, 0.0)]]
    traj = preprocess_paths(paths, num_points=16)

    saved = evalr.save_template("test", traj)
    assert saved is not None

    result = evalr.evaluate("test", traj)
    assert result["has_templates"] is True
    assert result["num_templates"] >= 1
    assert result["score"] is not None
    assert result["score"] <= 1.0
    assert result["score"] >= 0.0