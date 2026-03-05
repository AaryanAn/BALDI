import json
from pathlib import Path

import numpy as np

from trajectory.normalization import preprocess_paths
from evaluation.dtw import dtw_distance


class LetterEvaluator:
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _label_dir(self, label: str) -> Path:
        return self.root / label

    def get_trajectory(self, paths, num_points: int = 100):
        return preprocess_paths(paths, num_points=num_points)

    def save_template(self, label: str, traj: np.ndarray):
        label = label.strip()
        if not label:
            return None

        d = self._label_dir(label)
        d.mkdir(parents=True, exist_ok=True)

        existing = sorted(d.glob("*.npy"))
        idx = len(existing)
        fname = d / f"{idx}.npy"
        np.save(fname, traj)

        meta = {
            "label": label,
            "index": idx,
        }
        with open(d / f"{idx}.json", "w") as f:
            json.dump(meta, f)

        return fname

    def load_templates(self, label: str):
        d = self._label_dir(label)
        if not d.exists():
            return []

        out = []
        for path in sorted(d.glob("*.npy")):
            arr = np.load(path)
            out.append(arr)

        return out

    def evaluate(self, label: str, traj: np.ndarray):
        templates = self.load_templates(label)
        if not templates:
            return {
                "label": label,
                "score": None,
                "distance": None,
                "has_templates": False,
                "num_templates": 0,
            }

        dists = [dtw_distance(traj, t) for t in templates]
        best = min(dists)

        # smaller is better; be stricter by scaling distance
        # good matches should have low distance, random shapes should crash to ~0
        scale = 3.0
        raw = 1.0 - scale * best
        score = float(max(0.0, min(1.0, raw)))

        return {
            "label": label,
            "score": score,
            "distance": float(best),
            "has_templates": True,
            "num_templates": len(templates),
        }

