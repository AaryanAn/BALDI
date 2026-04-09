"""Generate DTW templates for the 28 Arabic letters from a system font.

Each letter is stored under templates/<unicode_char>/ using its isolated form,
which is what the user sees/draws when writing a single letter in isolation.
"""

from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

from evaluation.letters import LetterEvaluator
from trajectory.normalization import preprocess_paths

# All 28 Arabic letters — Unicode isolated forms, in abjad order
ARABIC_LETTERS_ORDERED: list[tuple[str, str]] = [
    ("\u0627", "alef"),
    ("\u0628", "ba"),
    ("\u062a", "ta"),
    ("\u062b", "tha"),
    ("\u062c", "jim"),
    ("\u062d", "ha"),
    ("\u062e", "kha"),
    ("\u062f", "dal"),
    ("\u0630", "dhal"),
    ("\u0631", "ra"),
    ("\u0632", "zay"),
    ("\u0633", "sin"),
    ("\u0634", "shin"),
    ("\u0635", "sad"),
    ("\u0636", "dad"),
    ("\u0637", "tah"),
    ("\u0638", "dha"),
    ("\u0639", "ain"),
    ("\u063a", "ghain"),
    ("\u0641", "fa"),
    ("\u0642", "qaf"),
    ("\u0643", "kaf"),
    ("\u0644", "lam"),
    ("\u0645", "mim"),
    ("\u0646", "nun"),
    ("\u0647", "heh"),
    ("\u0648", "waw"),
    ("\u064a", "ya"),
]

_FONT_CANDIDATES = [
    "/System/Library/Fonts/GeezaPro.ttc",
    "/System/Library/Fonts/Supplemental/Arabic Typesetting.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _find_arabic_font() -> str | None:
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _glyph_paths(ch: str, prop: FontProperties, size: float = 1.0) -> list[list[tuple[float, float]]]:
    tp = TextPath((0.0, 0.0), ch, size=size, prop=prop)
    verts = tp.vertices
    codes = tp.codes

    strokes: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for (x, y), code in zip(verts, codes):
        x, y = float(x), float(-y)  # flip Y so top-to-bottom matches drawing direction

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


def build_arabic_font_templates(overwrite: bool = False) -> list[str]:
    """Generate and save font-based templates for all 28 Arabic letters.

    Args:
        overwrite: If False (default), skip letters that already have a template.

    Returns:
        List of letter names that were successfully saved.
    """
    font_path = _find_arabic_font()
    if font_path is None:
        raise RuntimeError(
            "No Arabic-capable font found on this system. "
            "Install one of: " + ", ".join(_FONT_CANDIDATES)
        )

    prop = FontProperties(fname=font_path)
    src_root = Path(__file__).resolve().parent.parent
    templates_dir = src_root / "templates"
    evaluator = LetterEvaluator(templates_dir)

    saved: list[str] = []

    for ch, name in ARABIC_LETTERS_ORDERED:
        label_dir = templates_dir / ch
        if not overwrite and label_dir.exists() and list(label_dir.glob("*.npy")):
            print(f"  skip {name} ({ch}) — template exists")
            continue

        paths = _glyph_paths(ch, prop)
        if not paths:
            print(f"  warn {name} ({ch}) — no glyph paths from font")
            continue

        traj = preprocess_paths(paths, num_points=100)
        if traj.shape[0] == 0:
            print(f"  warn {name} ({ch}) — trajectory is empty after preprocessing")
            continue

        evaluator.save_template(ch, traj)
        print(f"  saved {name} ({ch})")
        saved.append(name)

    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Arabic letter templates from system font")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing templates")
    args = parser.parse_args()

    print(f"Building Arabic font templates (overwrite={args.overwrite}) ...")
    result = build_arabic_font_templates(overwrite=args.overwrite)
    print(f"Done — {len(result)} templates saved: {result}")
