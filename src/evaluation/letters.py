import json
from pathlib import Path

import numpy as np

from trajectory.normalization import preprocess_paths
from evaluation.dtw import dtw_distance


class LetterEvaluator:
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache = {}

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
        if label in self._cache:
            return self._cache[label]
        d = self._label_dir(label)
        if not d.exists():
            return []

        out = []
        for path in sorted(d.glob("*.npy")):
            arr = np.load(path)
            out.append(arr)

        self._cache[label] = out
        return out

    def list_labels(self):
        labels = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                labels.append(p.name)
        return labels

    def _score_from_dist(self, dist: float):
        # nonlinear mapping: tuned for air-writing trajectories
        k = 2.0
        return float(max(0.0, min(1.0, np.exp(-k * float(dist)))))

    def predict(
        self,
        traj: np.ndarray,
        labels: list[str] | None = None,
        top_k: int = 3,
        best_max_dist: float = 0.60,
        gap_min_dist: float = 0.015,
        collapse_case: bool = True,
    ):
        """
        Predict the most likely label by evaluating DTW against all templates.

        Returns:
          {
            "available": bool,
            "predicted_label": str|None,
            "confidence": float|None,   # 0..1
            "best_distance": float|None,
            "second_distance": float|None,
            "top": [{"label": str, "distance": float, "score": float}, ...]
          }
        """
        if traj is None or np.asarray(traj).ndim != 2 or len(traj) < 2:
            return {
                "available": False,
                "predicted_label": None,
                "confidence": None,
                "best_distance": None,
                "second_distance": None,
                "top": [],
            }

        if labels is None:
            labels = self.list_labels()
        if collapse_case:
            # Group labels by case-folded key (e.g. 'A' and 'a' compete inside the same bucket).
            # This makes "A vs a" not look like an ambiguity to the user.
            grouped = {}
            for lbl in labels:
                grouped.setdefault(lbl.lower(), []).append(lbl)
            label_groups = list(grouped.values())
        else:
            label_groups = [[lbl] for lbl in labels]

        per_label = []
        for group in label_groups:
            best_label = None
            best_dist = float("inf")
            for label in group:
                templates = self.load_templates(label)
                if not templates:
                    continue
                d = min(dtw_distance(traj, t) for t in templates)
                if d < best_dist:
                    best_dist = float(d)
                    best_label = label
            if best_label is not None:
                per_label.append((best_label, float(best_dist)))

        if not per_label:
            return {
                "available": False,
                "predicted_label": None,
                "confidence": None,
                "best_distance": None,
                "second_distance": None,
                "top": [],
            }

        per_label.sort(key=lambda x: x[1])
        best_label, best_dist = per_label[0]
        second_dist = per_label[1][1] if len(per_label) > 1 else float("inf")

        score_best = self._score_from_dist(best_dist)
        sep = float(max(0.0, second_dist - best_dist))
        k_sep = 30.0
        confidence = float(max(0.0, min(1.0, score_best * (1.0 - float(np.exp(-k_sep * sep))))))

        uncertain = (best_dist > best_max_dist) or (sep < gap_min_dist)
        predicted_label = None if uncertain else best_label

        top = []
        for label, dist in per_label[: max(1, int(top_k))]:
            top.append({"label": label, "distance": float(dist), "score": self._score_from_dist(dist)})

        return {
            "available": True,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "best_distance": float(best_dist),
            "second_distance": float(second_dist) if np.isfinite(second_dist) else None,
            "top": top,
        }

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

        score = self._score_from_dist(best)

        return {
            "label": label,
            "score": score,
            "distance": float(best),
            "has_templates": True,
            "num_templates": len(templates),
        }

