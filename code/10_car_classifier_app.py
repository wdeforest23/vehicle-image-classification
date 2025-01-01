import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenet
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

# Set page config with dark theme
st.set_page_config(
    page_title="Car Analysis App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the downloads directory path
downloads_dir = str(Path.home() / "Downloads")

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CONFIDENCE_THRESHOLD = 0.9

# Model paths and configurations
MODEL_CONFIGS = {
    'MobileNet Model': {
        'path': os.path.join(downloads_dir, 'mobilenet5_fine_tuned_model.keras'),
        'preprocess': preprocess_input_mobilenet
    },
    'ResNet50 Model': {
        'path': os.path.join(downloads_dir, 'car_classifier_resnet50_v2_final.h5'),
        'preprocess': preprocess_input_resnet50
    }
}

DETECTION_MODELS = {
    'YOLOv8 Car Detection': os.path.join(downloads_dir, 'yolov8_car_detection_ks (1).pt')
}

@st.cache_resource
def load_yolo_model(model_path):
    """Load and cache the YOLO model"""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {str(e)}")
        return None

def process_yolo_detection(image, model_path):
    """Process image with YOLO model and return annotated image and results"""
    yolo_model = load_yolo_model(model_path)
    if yolo_model is None:
        return None, None
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run inference
    results = yolo_model(img_array)
    
    # Filter boxes based on confidence threshold
    filtered_boxes = []
    for box in results[0].boxes:
        if float(box.conf[0]) >= CONFIDENCE_THRESHOLD:
            filtered_boxes.append(box)
    
    if not filtered_boxes:
        return img_array, None
    
    # Plot results on image (only for confident detections)
    results[0].boxes = filtered_boxes
    annotated_img = results[0].plot()
    
    return annotated_img, results[0]

@st.cache_data
def load_class_names():
    try:
        df = pd.read_csv(os.path.join(downloads_dir, 'class_names.csv'))
        return sorted(df['class_name'].tolist())
    except Exception as e:
        st.error(f"Error loading class names: {str(e)}")
        return None

@st.cache_resource
def load_models():
    models = {}
    for name, config in MODEL_CONFIGS.items():
        try:
            if config['path'].endswith('.keras'):
                models[name] = tf.keras.models.load_model(config['path'])
            else:
                models[name] = load_model(config['path'])
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
    return models

def preprocess_image(image, model_name):
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_name in MODEL_CONFIGS:
        img_array = MODEL_CONFIGS[model_name]['preprocess'](img_array)
    
    return img_array

def main():
    st.title("🚗 Car Analysis App")
    st.write("Upload an image of a car for detection or classification!")

    # Sidebar
    with st.sidebar:
        st.header("Mode Selection")
        app_mode = st.radio(
            "Choose Operation Mode",
            ["Object Detection", "Classification"]
        )
        
        if app_mode == "Object Detection":
            selected_detector = st.selectbox(
                "Choose detection model",
                list(DETECTION_MODELS.keys())
            )
        else:  # Classification mode
            selected_classifier = st.selectbox(
                "Choose classification model",
                list(MODEL_CONFIGS.keys())
            )
            if st.checkbox("Show available classes"):
                class_names = load_class_names()
                if class_names:
                    st.write("Available car classes:", len(class_names))
                    st.write(class_names[:10])

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            with col2:
                if app_mode == "Object Detection":
                    st.subheader("Detection Results")
                    detected_image, results = process_yolo_detection(
                        image, 
                        DETECTION_MODELS[selected_detector]
                    )
                    
                    if results is not None and len(results.boxes) > 0:
                        st.image(Image.fromarray(detected_image), use_container_width=True)
                        confident_detections = [box for box in results.boxes if float(box.conf[0]) >= CONFIDENCE_THRESHOLD]
                        st.write(f"Found {len(confident_detections)} cars with confidence ≥ {CONFIDENCE_THRESHOLD}")
                        for box in confident_detections:
                            conf = float(box.conf[0])
                            st.write(f"Confidence: {conf:.2f}")
                    else:
                        st.image(image, use_container_width=True)
                        st.write("No car found with confidence ≥ 0.9")
                
                else:  # Classification mode
                    st.subheader("Classification Results")
                    class_names = load_class_names()
                    if class_names:
                        with st.spinner("Analyzing car..."):
                            models = load_models()
                            classifier_model = models[selected_classifier]
                            processed_image = preprocess_image(image, selected_classifier)
                            predictions = classifier_model.predict(processed_image)
                            predictions = predictions[0]
                            top_3_idx = np.argsort(predictions)[-3:][::-1]

                            for idx, pred_idx in enumerate(top_3_idx):
                                car_name = class_names[pred_idx]
                                confidence = float(predictions[pred_idx]) * 100

                                st.markdown(f"""
                                <div style="padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; 
                                        background-color: rgba(255, 255, 255, 0.1);">
                                    <span style="color: white; font-size: 1.2rem;">
                                        {idx + 1}. {car_name}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.progress(confidence/100)
                                st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error("Error processing image")
            st.exception(e)

if __name__ == "__main__":
    main()