import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from evaluation.letters import LetterEvaluator
from trajectory.normalization import preprocess_paths


def test_autopredict_returns_label_when_clear_winner(tmp_path):
    ev = LetterEvaluator(tmp_path)

    a = preprocess_paths([[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]], num_points=60)
    b = preprocess_paths([[(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]], num_points=60)

    ev.save_template("A", a)
    ev.save_template("B", b)

    out = ev.predict(a, top_k=2)
    assert out["available"] is True
    assert out["predicted_label"] == "A"
    assert out["confidence"] is not None
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["top"]) == 2


def test_autopredict_uncertain_for_ambiguous(tmp_path):
    ev = LetterEvaluator(tmp_path)

    a = preprocess_paths([[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]], num_points=60)
    # Make B identical -> ambiguous (gap ~ 0)
    ev.save_template("A", a)
    ev.save_template("B", a)

    out = ev.predict(a, top_k=2, gap_min_dist=0.05)
    assert out["available"] is True
    assert out["predicted_label"] is None

