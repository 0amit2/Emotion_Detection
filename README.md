# Real-Time Emotion Detection System

### Project Overview
Developed a high-performance, real-time emotion recognition system that leverages Deep Learning and Computer Vision to identify human facial expressions via a live webcam feed. The application is built using the Streamlit framework for a responsive web interface and utilizes a Convolutional Neural Network (CNN) architecture for accurate classification.

### Technical Stack
* Language: Python
* Deep Learning Framework: TensorFlow / Keras
* Computer Vision: OpenCV (Haar Cascades for face detection)
* Web Framework: Streamlit
* Data Processing: NumPy

### Key Features
* Live Stream Processing: Captures and processes video frames in real-time with optimized latency.
* Face Localization: Uses Haar Cascade Classifiers to detect and isolate facial regions within a frame.
* Emotion Classification: Categorizes detected faces into seven distinct emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.
* Resource Management: Implemented robust session state management and hardware release protocols to ensure the camera hardware is properly de-allocated upon process termination.
* Model Optimization: Employs a pre-trained CNN architecture loaded via JSON for efficient inference.

### Architecture and Implementation
The system follows a modular pipeline:
1. Preprocessing: Converts incoming BGR frames to Grayscale for initial face detection.
2. Region of Interest (ROI) Extraction: Isolates the detected face and resizes it to 48x48 pixels.
3. Normalization: Scales pixel values to a [0, 1] range to improve model prediction accuracy.
4. Inference: Pass the ROI through the trained Keras model to obtain probability distributions.
5. Visualization: Overlays bounding boxes and classification labels onto the live video feed.

### Setup and Installation

1. Install dependencies:
pip install -r requirements.txt

2. Execution:
Ensure the model files (model_emotions.json and weights_emotions.hdf5) are present in the root directory, then run:
python -m streamlit run app.py

