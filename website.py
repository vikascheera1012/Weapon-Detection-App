import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np
import requests

st.title("Weapon Detection App")

# Model settings
MODEL_PATH = "today_weapon_model_best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1kZIT974w1I8i8or0kDf5EK1dSLsYNog0"

# 1. Download model from Google Drive if not available
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found. Downloading...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("✅ Model downloaded successfully!")

# 2. Load model
with st.spinner("Loading model..."):
    model = YOLO(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# 3. Input type selection
input_type = st.radio("Select Input Type", ("Image", "Video", "Webcam"))

# 4. Functions
def run_inference_on_image(img):
    results = model(img)
    annotated_img = results[0].plot()
    return annotated_img

def run_inference_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        stframe.image(annotated_frame, channels="BGR")
    cap.release()

# 5. Image Upload
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        annotated_img = run_inference_on_image(img)
        st.image(annotated_img, caption="Detection Result", use_column_width=True)

# 6. Video Upload
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        run_inference_on_video(tfile.name)
        os.unlink(tfile.name)

# 7. Webcam
elif input_type == "Webcam":
    st.info("Click Start to open webcam and run detection.")
    start = st.button("Start Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to open webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            stframe.image(annotated_frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
