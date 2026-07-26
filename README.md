# CNN FER2013 Emotion Recognition Web App

This project trains a CNN model on FER2013 and serves a web UI for facial expression prediction.

## Features
- Train CNN on FER2013 grayscale images (48x48)
- Predict 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Web app with 2 input modes: image upload and webcam capture
- Display labels in Vietnamese or English

## Project Structure

```text
CNN_FER2013_WebApp/
  data/
    fer2013.csv                # You place dataset here
  models/
  src/
    train.py                   # Train model
    app.py                     # Streamlit web app
    utils.py                   # Preprocess and decode helpers
  requirements.txt
  README.md
```

## 1) Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2) Prepare FER2013 Data
Download FER2013 csv and place it at:

```text
data/fer2013.csv
```

Required columns:
- emotion
- pixels
- Usage

## 3) Train Model

```bash
python src/train.py --csv data/fer2013.csv --epochs 30 --batch-size 64
```

After training, model files are generated in `models/`.

## 4) Run Web App

```bash
streamlit run src/app.py
```

Then open the local URL shown by Streamlit.

Mini app (simple upload + predict):

```bash
streamlit run src/app_mini.py
```

In the app:
- Select display language (`vi` or `en`)
- Choose input source (`Tai anh` or `Camera`)
- Review prediction, confidence, and class probability chart

## Notes
- If the model is missing, the app will show a warning.
- For better inference, crop the image around the face before upload.

## 5) Predict From CLI

```bash
python src/predict.py --image path/to/your_face_image.jpg
```
