# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model once
@st.cache_resource
def load_model():
    model_path = "./dog_breed_classifier_model.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# List of dog breed labels (same as training order)
labels = ["Beagle", "Boxer", "Bulldog", "Dachshund", "German_Shepherd", 
          "Golden_Retriever", "Labrador_Retriever", "Poodle", "Rottweiler", "Yorkshire_Terrier"]

# Streamlit App UI
st.title("🐶 Dog Breed Classifier")
st.markdown("Upload a picture of a dog, and the model will predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((150, 150))
    image_array = np.array(img_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index]

    # Show result
    st.markdown(f"### 🐾 Predicted Breed: **{predicted_label}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")
