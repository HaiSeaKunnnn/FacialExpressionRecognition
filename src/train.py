from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import EMOTION_LABELS


def parse_pixels(pixel_series: pd.Series) -> np.ndarray:
    pixel_arrays = [np.fromstring(p, sep=" ", dtype=np.float32) for p in pixel_series]
    x = np.array(pixel_arrays)
    x = x.reshape((-1, 48, 48, 1))
    x /= 255.0
    return x


def build_model(input_shape=(48, 48, 1), n_classes=7) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_fer2013(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {"emotion", "pixels", "Usage"}.issubset(df.columns):
        raise ValueError("FER2013 csv must have columns: emotion, pixels, Usage")

    x_all = parse_pixels(df["pixels"])
    y_all = tf.keras.utils.to_categorical(df["emotion"].values, num_classes=7)

    train_mask = df["Usage"] == "Training"
    val_mask = df["Usage"].isin(["PublicTest", "PrivateTest"])

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    return (x_train, y_train), (x_val, y_val)


def load_from_directory(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected folder structure: data/train/<class> and data/test/<class>"
        )

    class_order = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",label_mode="categorical",color_mode="grayscale",
        class_names=class_order,image_size=(48, 48),
        batch_size=batch_size,shuffle=True,seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        class_names=class_order,
        image_size=(48, 48),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def main():
    parser = argparse.ArgumentParser(description="Train CNN on FER2013")
    parser.add_argument("--csv", type=str, default=None, help="Path to fer2013.csv")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data dir containing train/ and test/ folders",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-output", type=str, default="models/fer2013_emotion_cnn.keras")
    parser.add_argument("--label-output", type=str, default="models/label_map.json")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    data_dir = Path(args.data_dir)
    model_output = Path(args.model_output)
    label_output = Path(args.label_output)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    label_output.parent.mkdir(parents=True, exist_ok=True)

    class_names = None
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        train_data, val_data, class_names = load_from_directory(data_dir, args.batch_size)
        print(f"Loaded directory dataset from: {data_dir}")
        print(f"Classes: {class_names}")
    elif csv_path and csv_path.exists():
        (x_train, y_train), (x_val, y_val) = load_fer2013(csv_path)
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        print(f"Train shape: {x_train.shape}, Val shape: {x_val.shape}")
    else:
        raise FileNotFoundError(
            "No valid dataset found. Provide data/train+data/test or --csv path."
        )

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output), monitor="val_accuracy", save_best_only=True
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(val_data, verbose=0)
    print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

    label_map = {}
    if class_names:
        for idx, name in enumerate(class_names):
            label_map[idx] = name
    else:
        label_map = EMOTION_LABELS

    with label_output.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"Model saved to: {model_output}")
    print(f"Label map saved to: {label_output}")


if __name__ == "__main__":
    main()
