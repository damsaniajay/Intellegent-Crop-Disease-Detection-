import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained model/crop_disease_detection_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, img_array, class_indices):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence_score = predictions[0][predicted_class_index]
    return predicted_class_name, confidence_score

# Streamlit App
st.title('Plant Disease Classifier')

# Upload and preprocess the image only once
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    img_array = load_and_preprocess_image(uploaded_image)
    st.session_state.img_array = img_array
    st.session_state.image_uploaded = True

# Display tabs for Identification and Visualization side by side
col1, col2 = st.columns(2)
with col1:
    if st.button('Identification'):
        st.session_state.tab_selected = 'Identification'
with col2:
    if st.button('Visualization'):
        st.session_state.tab_selected = 'Visualization'

selected_tab = st.session_state.get('tab_selected', 'Identification')

if st.session_state.get('image_uploaded', False):
    if selected_tab == 'Identification':
        st.header('Plant Disease Identification')
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=False)
        predicted_class, confidence_score = predict_image_class(model, img_array, class_indices)
        st.write(f'Prediction: {predicted_class} ({confidence_score:.2f} confidence)')

    elif selected_tab == 'Visualization':
        st.header('Confidence Scores Visualization')
        plt.figure(figsize=(12, 6))  # Smaller graph size
        plt.bar(class_indices.values(), model.predict(img_array)[0])
        plt.xlabel('Class')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=90, ha='right')
        plt.title('Confidence Scores for Predicted Classes')
        st.pyplot(plt)





