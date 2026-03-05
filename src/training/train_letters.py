import argparse
from pathlib import Path

import keras

from models.trajectory_classifier import build_trajectory_classifier
from training.letters_dataset import build_tf_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="Root folder with label subdirs of .npy trajectories")
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--model-out", type=str, default="models/letters_english.keras")
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.data_root).resolve()
    ds, idx_to_label, _ = build_tf_dataset(root, seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True)

    num_classes = len(idx_to_label)
    if num_classes == 0:
        raise SystemExit(f"No label folders found under {root}")

    total = 0
    for _ in ds:
        total += 1
    if total == 0:
        raise SystemExit(f"No trajectories found under {root}")

    steps = total
    val_steps = max(1, int(steps * args.val_split))
    train_steps = max(1, steps - val_steps)

    train_ds = ds.take(train_steps)
    val_ds = ds.skip(train_steps).take(val_steps)

    model = build_trajectory_classifier(num_classes=num_classes, seq_len=args.seq_len)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
    )

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)


if __name__ == "__main__":
    main()

