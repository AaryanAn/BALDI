"""
Render a normalized 2D trajectory (Nx2 array) to a grayscale image for the image classifier.
Points are assumed in a normalized space (e.g. roughly [-1, 1] or [0, 1]); we scale to image size.
"""
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def trajectory_to_image(
    points: np.ndarray,
    size: int = 64,
    stroke_width: int = 2,
    margin: float = 0.1,
) -> np.ndarray:
    """
    Draw trajectory on a size x size grayscale image (0 = white bg, 255 = black stroke).
    points: (N, 2) float array.
    """
    if points is None or len(points) < 2:
        out = 255 * np.ones((size, size), dtype=np.uint8)
        return out

    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    scale = (1.0 - 2 * margin) * min(size / span_x, size / span_y)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    px = (size / 2 + (x - cx) * scale).astype(np.int32)
    py = (size / 2 - (y - cy) * scale).astype(np.int32)  # y flip

    out = 255 * np.ones((size, size), dtype=np.uint8)
    for i in range(len(px) - 1):
        _draw_line(out, px[i], py[i], px[i + 1], py[i + 1], stroke_width)
    return out


def _draw_line(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, w: int):
    h, ww = canvas.shape
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for t in np.linspace(0, 1, steps + 1):
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        for dx in range(-w, w + 1):
            for dy in range(-w, w + 1):
                if dx * dx + dy * dy <= w * w:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < ww and 0 <= ny < h:
                        canvas[ny, nx] = 0


def save_trajectory_image(points: np.ndarray, path: str | Path, size: int = 64, **kwargs):
    """Render trajectory to image and save as PNG."""
    if Image is None:
        raise ImportError("PIL is required for save_trajectory_image")
    img = trajectory_to_image(points, size=size, **kwargs)
    Image.fromarray(img).save(path)
