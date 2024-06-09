import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import streamlit as st
import cv2
import gdown
import os

# Function to download the model from Google Drive
def download_model(file_id, output_path):
    try:
        if not os.path.exists(output_path):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=True)
    except Exception as e:
        st.error(f"Error downloading the model: {e}")

# Download model if it does not exist locally
file_id = '17-dxaC04oO95hMExUC_IOoPO0RaRlfkF'
model_path = 'Nadam_TTS_Epoch50.h5'
download_model(file_id, model_path)

# Load the model
model = load_model(model_path)

def get_img_array(img, size):
    img = img.resize(size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    
    jet_heatmap = Image.fromarray(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    fig, ax = plt.subplots()
    ax.imshow(superimposed_img)
    ax.axis('off')
    st.pyplot(fig)

st.title("Malaria Detection App with Heatmap")
st.write("Upload an image to detect malaria and see the heatmap.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((256, 256))  # Resize uploaded image
    img_array = get_img_array(img, size=(128, 128))
    
    last_conv_layer_name = "conv2d_35"
    classifier_layer_names = [
        "max_pooling2d_35", "batch_normalization_51", "dropout_51", "flatten_8",
        "dense_24", "batch_normalization_52", "dropout_52", "dense_25", 
        "batch_normalization_53", "dropout_53", "dense_26"
    ]
    
    with st.spinner('Classifying...'):
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        prediction = model.predict(img_array)
        malaria_probability = prediction[0][0] * 100
        result = 'Malaria' if malaria_probability > 50 else 'No Malaria'
        
        st.write(f"Prediction: {result}")
        st.write(f"Probability: {malaria_probability:.2f}%")
        
        alpha = st.slider('Adjust Heatmap Intensity', 0.0, 1.0, 0.4)
        save_and_display_gradcam(img, heatmap, alpha=alpha)

        # Add description of the heatmap
        st.write("Heatmap Description: The red areas indicate regions considered important by the model for malaria prediction.")
