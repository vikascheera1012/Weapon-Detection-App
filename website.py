import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
import numpy as np

st.title("Weapon Detection App")

# 1. Load your local YOLO .pt model (change path to your actual file)
model_path = r"C:\Users\praveen sana\OneDrive\Desktop\vikas\vikass\vikas\practise\today_weapon_data\today_weapon_model_best.pt"
with st.spinner("Loading model..."):
    model = YOLO(model_path)
st.success("Model loaded successfully!")

# 2. Choose input type
input_type = st.radio("Select Input Type", ("Image", "Video", "Webcam"))

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
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        annotated_frame = results[0].plot()
        # Convert back to BGR for display with OpenCV (if needed)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        stframe.image(annotated_frame, channels="BGR")
    cap.release()

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        annotated_img = run_inference_on_image(img)
        st.image(annotated_img, caption="Detection Result", use_column_width=True)

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save to temp file to read with OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        run_inference_on_video(tfile.name)
        os.unlink(tfile.name)

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
            
            # Break loop if user closes Streamlit or presses some key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
