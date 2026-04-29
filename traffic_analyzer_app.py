import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import time
import tempfile 
import os 
from collections import deque # Optional: for smoother FPS calculation display

# --- 1. CONFIGURATION & SETUP ---

TARGET_CLASSES = [2, 5, 7]  # Car, Bus, Truck
LOW_THRESHOLD = 5
MEDIUM_THRESHOLD = 15

# Use Streamlit's caching mechanism to load the large YOLO model only ONCE.
@st.cache_resource 
def load_model():
    """Loads the YOLOv8n model weights."""
    return YOLO('yolov8n.pt')

# Load the model outside the functions for global access
model = load_model()

# --- 2. CORE PROCESSING LOGIC (Adapted for Streamlit) ---

def process_video_stream(video_path, display_width=800, inference_width=0):
    """
    Handles video processing, object detection, counting, and display for Streamlit.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Initialize Streamlit placeholders for display and metrics
    st.markdown("### Processed Video Feed")
    frame_placeholder = st.empty()
    metric_placeholder = st.empty()
    
    # Simple FPS tracking setup
    fps_history = deque(maxlen=5)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {video_path}")
        return

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Optionally resize frame for faster inference (preserve aspect ratio)
        frame_for_inference = frame
        if inference_width and inference_width > 0 and inference_width != frame.shape[1]:
            h, w = frame.shape[:2]
            scale = inference_width / float(w)
            new_h = max(1, int(h * scale))
            frame_for_inference = cv2.resize(frame, (inference_width, new_h))

        # 1. Detection
        results = model(frame_for_inference, verbose=False)
        annotated_frame = results[0].plot()

        # 2. Counting and Classification
        detections = results[0].boxes.cls.tolist()
        vehicle_count = 0
        
        for class_id in detections:
            if int(class_id) in TARGET_CLASSES:
                vehicle_count += 1

        # 3. Density Classification
        if vehicle_count <= LOW_THRESHOLD:
            density = "LOW"
            color = (0, 255, 0) # Green
        elif vehicle_count <= MEDIUM_THRESHOLD:
            density = "MEDIUM"
            color = (0, 255, 255) # Yellow
        else:
            density = "HIGH"
            color = (0, 0, 255) # Red
        
        # 4. Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # 5. Overlay Data
        text = f"Count: {vehicle_count} | Density: {density} | FPS: {avg_fps:.2f}"
        cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)
        
        # 6. Streamlit Display (RGB conversion is mandatory)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # Use explicit width to allow larger/smaller display controlled by the user
        frame_placeholder.image(frame_rgb, channels="RGB", width=display_width)

    cap.release()
    st.success("Video processing complete!")

# --- 3. STREAMLIT APPLICATION ENTRY POINT ---

def main_app():
    st.title("🚦 Real-Time Traffic Density Analyzer")
    st.markdown("---")
    # Sidebar controls for display size and options
    st.sidebar.markdown("**Display Options**")
    display_width = st.sidebar.slider("Display width (px)", min_value=320, max_value=1920, value=800, step=10)
    show_original = st.sidebar.checkbox("Show uploaded original video", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Performance**")
    inference_width = st.sidebar.slider("Resize width for inference (px, 0 = no resize)", min_value=0, max_value=1280, value=0, step=10)

    video_file = st.file_uploader("Upload a traffic video file (.mp4) for analysis:", type=['mp4'])

    if video_file is not None:
        # Optionally show the uploaded file first (Streamlit's built-in player)
        if show_original:
            st.markdown("### Original Uploaded Video")
            st.video(video_file)

        # Create a temporary file to save the uploaded stream
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(video_file.read())
            video_path = tfile.name

        # Process the video stream with chosen display width and inference resize
        process_video_stream(video_path, display_width=display_width, inference_width=inference_width)

        # Clean up the temporary file after processing
        os.unlink(video_path)

if __name__ == "__main__":
    main_app()