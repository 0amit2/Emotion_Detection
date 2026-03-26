import cv2
import numpy as np
import streamlit as st
import os
from tensorflow.keras.models import model_from_json

# --- Page Setup ---
st.set_page_config(page_title="Real-time Emotion Detector", layout="centered")
st.title('Real-time Emotion Detection')

# --- Path Helper ---
def get_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# --- Load Model (Cached) ---
@st.cache_resource
def load_emotion_model():
    try:
        with open(get_path('model_emotions.json'), 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(get_path("weights_emotions.hdf5"))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize Model and Face Detector
emotion_model = load_emotion_model()
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# --- Session State Management ---
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# --- UI ---
col1, col2 = st.columns(2)

with col1:
    if st.button('Start Detection', use_container_width=True):
        st.session_state.run_camera = True

with col2:
    if st.button('Stop Detection', use_container_width=True):
        st.session_state.run_camera = False
        st.rerun()  

# Placeholder for the video feed
FRAME_WINDOW = st.image([])

# --- Real-Time Loop ---
if st.session_state.run_camera:
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        st.error("Cannot access webcam. Please check permissions.")
        st.session_state.run_camera = False
    else:
        try:
            while st.session_state.run_camera:
                success, img = cam.read()
                if not success:
                    st.warning("Failed to grab frame.")
                    break

                # 1. Processing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_model.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_roi = img[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = face_roi.astype("float32") / 255.0
                    face_roi = np.expand_dims(face_roi, axis=0)

                    prediction = emotion_model.predict(face_roi, verbose=0)
                    emotion = emotion_labels[np.argmax(prediction)]

                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, emotion, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 2. Display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(img_rgb)

        finally:
           
            cam.release()
            cv2.destroyAllWindows()
            FRAME_WINDOW.empty()
           
else:
    st.info(" Click 'Start' to begin detection.")