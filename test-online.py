import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# ID file Google Drive
file_id = '17-dxaC04oO95hMExUC_IOoPO0RaRlfkF'
model_path = 'Nadam_TTS_Epoch50.h5'

# Unduh model dari Google Drive
if not os.path.exists(model_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Function to make predictions
def model_predict(img, model):
    img = img.resize((128, 128))  # Sesuaikan ukuran gambar sesuai dengan input model Anda
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Streamlit interface
st.title("Malaria Detection App")
st.write("Upload an image to detect malaria.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.write("Classifying...")
        prediction = model_predict(img, model)
        malaria_probability = prediction[0][0] * 100  # Convert to percentage
        result = 'Malaria' if malaria_probability > 50 else 'No Malaria'
        
        st.write(f"Prediction: {result}")
        st.write(f"Probability of Malaria: {malaria_probability:.2f}%")
