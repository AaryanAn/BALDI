import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from trajectory.normalization import flatten_paths, normalize, preprocess_paths, resample


def test_flatten_paths_empty():
    out = flatten_paths([])
    assert out.shape == (0, 2)


def test_flatten_paths_basic():
    paths = [[(0, 0), (1, 1)], [(2, 2)]]
    out = flatten_paths(paths)
    assert out.shape == (3, 2)
    assert np.allclose(out[0], [0, 0])
    assert np.allclose(out[-1], [2, 2])


def test_resample_length():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    out = resample(pts, num_points=50)
    assert out.shape == (50, 2)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[-1], [1.0, 0.0])


def test_normalize_center_and_scale():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    norm = normalize(pts)
    mean = norm.mean(axis=0)
    assert np.allclose(mean, [0.0, 0.0], atol=1e-5)
    max_norm = np.max(np.linalg.norm(norm, axis=1))
    assert np.isclose(max_norm, 1.0, atol=1e-5)


def test_preprocess_paths_round_trip():
    paths = [[(0, 0), (10, 0), (20, 0)]]
    out = preprocess_paths(paths, num_points=32)
    assert out.shape == (32, 2)