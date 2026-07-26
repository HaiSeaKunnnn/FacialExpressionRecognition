from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image

EMOTION_LABELS: Dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

EMOTION_LABELS_VI: Dict[int, str] = {
    0: "Tức giận",
    1: "Ghê tởm",
    2: "Sợ hãi",
    3: "Vui vẻ",
    4: "Buồn",
    5: "Ngạc nhiên",
    6: "Bình thường",
}


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert input image to FER2013 format: 48x48 grayscale, normalized."""
    img = image.convert("L").resize((48, 48))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr


def decode_prediction(probabilities: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
    """Return top class and class probabilities."""
    probs = probabilities.flatten()
    class_id = int(np.argmax(probs))
    label = EMOTION_LABELS.get(class_id, f"Class_{class_id}")
    confidence = float(probs[class_id])
    details = {
        EMOTION_LABELS.get(i, f"Class_{i}"): float(score)
        for i, score in enumerate(probs)
    }
    return label, confidence, details


def emotion_label_map(language: str = "en") -> Dict[int, str]:
    if language.lower() == "vi":
        return EMOTION_LABELS_VI
    return EMOTION_LABELS
