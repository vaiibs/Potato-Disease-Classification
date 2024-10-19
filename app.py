import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

# Load the trained model and define class names
model = load_model('potato_model.keras')
class_names = ['Potato Early Blight', 'Potato Late Blight', 'Potato Healthy']
img_size = 180

# Function to classify an uploaded image
def classify_image(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    # Predict class probabilities
    predictions = model.predict(input_image_exp_dim)
    predicted_class = class_names[np.argmax(predictions)]
    confidence_percent = np.max(predictions) * 100

    return predicted_class, confidence_percent

# Streamlit app layout
def main():
    st.title('Potato Disease Classification')
    st.write('Upload a potato leaf image to classify the disease.')

    # File upload and classification
    uploaded_file = st.file_uploader('Choose a potato leaf image...', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = tf.keras.utils.load_img(uploaded_file, target_size=(img_size, img_size))
        st.image(image, caption='Uploaded potato leaf image.')
        st.write('Classifying...')

        # Classify the image and display results
        predicted_class, confidence = classify_image(uploaded_file)
        st.write(f'The potato leaf is classified as **{predicted_class}** with **{confidence:.2f}%** confidence.')

if __name__ == '__main__':
    main()
