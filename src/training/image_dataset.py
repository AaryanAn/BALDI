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


def _iter_images(root: Path, label_to_idx: Dict[str, int], img_size: int):
    for label, idx in label_to_idx.items():
        label_dir = root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.png")) + sorted(label_dir.glob("*.jpg")):
            try:
                raw = tf.io.read_file(str(path))
                img = tf.io.decode_image(raw, channels=1, expand_animations=False)
                img = tf.image.resize(img, [img_size, img_size])
                img = tf.cast(img, tf.float32) / 255.0
                if img.shape.ndims == 2:
                    img = tf.expand_dims(img, -1)
                yield img, np.int32(idx)
            except Exception:
                continue


def build_tf_dataset(
    root: Path,
    img_size: int,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, Dict[int, str], Dict[str, int]]:
    idx_to_label, label_to_idx = load_label_map(root)

    def gen():
        yield from _iter_images(root, label_to_idx, img_size)

    output_signature = (
        tf.TensorSpec(shape=(img_size, img_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, idx_to_label, label_to_idx
