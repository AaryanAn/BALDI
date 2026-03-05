import argparse
import json
from pathlib import Path

from models.image_classifier import build_image_classifier
from training.image_dataset import build_tf_dataset


def main():
    p = argparse.ArgumentParser(description="Train image classifier on letters (e.g. Kaggle ground truth)")
    p.add_argument("--data-root", type=str, default="../data/letters_images", help="Root with label subdirs of images (default: repo data/letters_images)")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--model-out", type=str, default="models/letters_english_image.keras")
    p.add_argument("--log-file", type=str, default=None, help="Write training summary to this file (e.g. ../logs/training_image_letters.log)")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = (root / args.data_root).resolve()
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}. Run scripts/convert_kaggle_to_images.py first.")

    ds, idx_to_label, _ = build_tf_dataset(
        data_root, img_size=args.img_size, batch_size=args.batch_size, shuffle=True
    )
    num_classes = len(idx_to_label)
    if num_classes == 0:
        raise SystemExit(f"No label folders under {data_root}")

    n_batches = sum(1 for _ in ds)
    if n_batches == 0:
        raise SystemExit(f"No images found under {data_root}")

    val_batches = max(1, n_batches // 10)
    train_ds = ds.skip(val_batches)
    val_ds = ds.take(val_batches)

    model = build_image_classifier(num_classes=num_classes, img_size=args.img_size)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    out_path = Path(args.model_out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    label_path = out_path.with_suffix(".labels.json")
    with open(label_path, "w") as f:
        json.dump(idx_to_label, f)
    print("Saved", out_path, "and", label_path)

    if args.log_file:
        log_path = (root / args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("training_image_letters\n")
            f.write(f"data_root={data_root}\n")
            f.write(f"num_classes={num_classes}\n")
            f.write(f"epochs={args.epochs}\n")
            f.write(f"batch_size={args.batch_size}\n")
            last = history.history
            if "loss" in last:
                f.write(f"final_loss={last['loss'][-1]:.6f}\n")
            if "accuracy" in last:
                f.write(f"final_accuracy={last['accuracy'][-1]:.6f}\n")
            if "val_loss" in last:
                f.write(f"final_val_loss={last['val_loss'][-1]:.6f}\n")
            if "val_accuracy" in last:
                f.write(f"final_val_accuracy={last['val_accuracy'][-1]:.6f}\n")
            f.write(f"model_saved={out_path}\n")
        print("Log written to", log_path)


if __name__ == "__main__":
    main()
