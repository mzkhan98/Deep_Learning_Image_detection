import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('my_model.h5', compile=False)

# Define the list of dog breeds your model can identify
dog_breeds = ["beagle", "bernese_mountain_dog", "doberman", "labrador_retriever", "siberian_husky"]

def process_image(image_path):
    """
    Loads an image and processes it to be input to the model.
    Adjust depending on your specific model input requirements.
    """
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Replace with your model's input size
    img = np.array(img) / 255.0  # Scale pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Model expects batches of images
    return img

def predict_breed(image_path):
    """
    Predicts the dog breed for a given image path.
    """
    img = process_image(image_path)
    predictions = model.predict(img)
    predicted_breed = dog_breeds[np.argmax(predictions)]
    return predicted_breed

st.title("Dog Breed Classifier")
st.write("Upload an image and the model will predict the breed of the dog.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    label = predict_breed(uploaded_file)
    st.write(f'The breed of the dog in the image is a {label}.')