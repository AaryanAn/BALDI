from pathlib import Path
from string import ascii_letters

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

from evaluation.letters import LetterEvaluator
from trajectory.normalization import preprocess_paths


def _glyph_paths(ch: str, size: float = 1.0):
    tp = TextPath((0.0, 0.0), ch, size=size)
    verts = tp.vertices
    codes = tp.codes

    strokes = []
    current = []

    for (x, y), code in zip(verts, codes):
        x = float(x)
        y = float(-y)

        if code == MplPath.MOVETO:
            if current:
                strokes.append(current)
            current = [(x, y)]
        elif code in (MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE4):
            current.append((x, y))
        elif code == MplPath.CLOSEPOLY:
            if current:
                current.append(current[0])
                strokes.append(current)
                current = []

    if current:
        strokes.append(current)

    return strokes


def build_font_templates(chars: str | None = None):
    if chars is None:
        chars = ascii_letters

    src_root = Path(__file__).resolve().parent.parent
    templates_dir = src_root / "templates"
    evaluator = LetterEvaluator(templates_dir)

    for ch in chars:
        paths = _glyph_paths(ch)
        if not paths:
            continue

        traj = preprocess_paths(paths, num_points=100)
        if traj.shape[0] == 0:
            continue

        evaluator.save_template(ch, traj)


if __name__ == "__main__":
    build_font_templates()

