from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import EMOTION_LABELS, preprocess_image

MODEL_PATH = Path("models/fer2013_emotion_cnn.keras")
LABEL_MAP_PATH = Path("models/label_map.json")

st.set_page_config(page_title="FER2013 Mini App", page_icon=":)" , layout="centered")
st.title("FER2013 Mini Web App")
st.caption("Upload one face image and get emotion prediction.")


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)


def load_label_map(path: Path) -> dict[int, str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}
    return EMOTION_LABELS


def top_k(probabilities: np.ndarray, labels: dict[int, str], k: int = 3):
    probs = probabilities.flatten()
    idxs = np.argsort(probs)[::-1][:k]
    return [(labels.get(int(i), f"class_{int(i)}"), float(probs[i])) for i in idxs]


model = load_model(MODEL_PATH)
labels = load_label_map(LABEL_MAP_PATH)

if model is None:
    st.error("Model not found. Train first: python src/train.py --data-dir data")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Predict", type="primary"):
        x = preprocess_image(image)
        probs = model.predict(x, verbose=0)

        best_idx = int(np.argmax(probs.flatten()))
        best_label = labels.get(best_idx, f"class_{best_idx}")
        best_conf = float(probs.flatten()[best_idx])

        st.subheader(f"Prediction: {best_label}")
        st.write(f"Confidence: {best_conf * 100:.2f}%")

        st.markdown("Top-3 probabilities")
        top3 = top_k(probs, labels, k=3)
        for name, score in top3:
            st.write(f"- {name}: {score * 100:.2f}%")

        chart_data = {labels.get(i, f"class_{i}"): float(v) for i, v in enumerate(probs.flatten())}
        st.bar_chart(chart_data)
