from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def load_label_map(root: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    labels: List[str] = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            labels.append(d.name)

    idx_to_label = {i: label for i, label in enumerate(labels)}
    label_to_idx = {label: i for i, label in idx_to_label.items()}
    return idx_to_label, label_to_idx


def _iter_trajectories(root: Path, label_to_idx: Dict[str, int], seq_len: int):
    for label, idx in label_to_idx.items():
        label_dir = root / label
        if not label_dir.exists():
            continue

        for path in sorted(label_dir.glob("*.npy")):
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue

            if arr.shape[0] != seq_len:
                if arr.shape[0] < 2:
                    continue

                x = np.linspace(0.0, 1.0, arr.shape[0])
                target = np.linspace(0.0, 1.0, seq_len)
                resampled = []
                for dim in range(2):
                    resampled_dim = np.interp(target, x, arr[:, dim])
                    resampled.append(resampled_dim)
                arr = np.stack(resampled, axis=1).astype(np.float32)

            yield arr.astype(np.float32), np.int32(idx)


def build_tf_dataset(
    root: Path,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, Dict[int, str], Dict[str, int]]:
    idx_to_label, label_to_idx = load_label_map(root)

    def gen():
        yield from _iter_trajectories(root, label_to_idx, seq_len)

    output_signature = (
        tf.TensorSpec(shape=(seq_len, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, idx_to_label, label_to_idx

