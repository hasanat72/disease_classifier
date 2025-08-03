import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('tomato_leaf_classifier.h5')

# Define the image size
img_height, img_width = 128, 128

# Define the class names (make sure these match the order in your training data)
# You might need to get these from train_generator.class_names if available
class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy'] # Replace with actual class names

st.title("Tomato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    img_array = tf.keras.utils.img_to_array(image.resize((img_height, img_width)))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get the predicted class and confidence
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    confidence = 100 * np.max(score)

    # Display the prediction
    st.write(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
