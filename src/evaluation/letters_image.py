"""
Use the Kaggle-trained image classifier to score a trajectory: render trajectory to image,
run through the model, return predicted letter and confidence. Use this as ground-truth
style evaluation (model trained on Kaggle English handwritten characters).
"""
import json
from pathlib import Path

import numpy as np

# Optional Keras; fail gracefully if not installed
try:
    import keras
except ImportError:
    keras = None


def _load_model_and_labels(model_path: Path, label_path: Path):
    if not model_path.exists() or not label_path.exists():
        return None, None
    if keras is None:
        return None, None
    model = keras.models.load_model(model_path)
    with open(label_path) as f:
        idx_to_label = json.load(f)
    # JSON keys are strings; convert to int for indexing
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    return model, idx_to_label


def predict_from_trajectory(
    traj: np.ndarray,
    model_path: str | Path | None = None,
    img_size: int = 64,
):
    """
    Render trajectory to image and run through the image classifier.
    Returns dict with predicted_label, confidence, and available=True/False.
    """
    from utils.render_trajectory import trajectory_to_image

    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "models" / "letters_english_image.keras"
    model_path = Path(model_path)
    label_path = model_path.with_suffix(".labels.json")

    model, idx_to_label = _load_model_and_labels(model_path, label_path)
    if model is None or idx_to_label is None:
        return {"available": False, "predicted_label": None, "confidence": None}

    img = trajectory_to_image(traj, size=img_size)
    # Model expects (1, H, W, 1), float normalized 0-1
    x = (255 - img).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))
    logits = model(x, training=False)
    probs = np.squeeze(logits)
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    predicted_label = idx_to_label.get(idx, "?")

    return {
        "available": True,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_probs": {idx_to_label.get(i, "?"): float(probs[i]) for i in range(len(probs))},
    }
