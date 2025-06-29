import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import requests
import time

st.set_page_config(page_title="Weapon Detection", layout="wide")
st.title("🔫 Weapon Detection App (Image | Video | Webcam)")

# MODEL CONFIG
MODEL_PATH = "today_weapon_model_best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kZIT974w1I8i8or0kDf5EK1dSLsYNog0"

# 🔽 Download model if not exists
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Downloading...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("✅ Model downloaded successfully!")

# 🧠 Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

with st.spinner("Loading model..."):
    model = load_model()
st.success("✅ Model loaded!")

# 📥 Input selection
input_type = st.radio("Select Input Type", ("Image", "Video", "Webcam"))

# 🖼️ Image input
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        results = model.predict(img)
        st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

# 🎞️ Video input
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            results = model.predict(frame, stream=True)
            for r in results:
                annotated_frame = r.plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
                break
            time.sleep(0.03)

        cap.release()
        os.unlink(tfile.name)

# 📷 Webcam input
elif input_type == "Webcam":
    st.info("Click 'Start Webcam' to begin detection from your camera.")
    start = st.button("Start Webcam")

    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read from webcam.")
                break

            frame = cv2.resize(frame, (640, 360))
            results = model.predict(frame, stream=True)
            for r in results:
                annotated_frame = r.plot()
                stframe.image(annotated_frame, channels="BGR", use_column_width=True)
                break

            time.sleep(0.03)

        cap.release()
