from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from PIL import Image

from utils import decode_prediction, preprocess_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict facial emotion from one image")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        default="models/fer2013_emotion_cnn.keras",
        type=str,
        help="Path to trained model",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    image = Image.open(image_path)
    x = preprocess_image(image)

    probabilities = model.predict(x, verbose=0)
    label, confidence, details = decode_prediction(probabilities)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Class probabilities:")
    for name, score in details.items():
        print(f"- {name}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
