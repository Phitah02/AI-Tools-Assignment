import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

model = load_model()

# Title
st.title("MNIST Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and get the prediction!")

# Drawing canvas
st.subheader("Or draw a digit below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Prediction from uploaded file
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display prediction
    st.write(f"**Predicted Digit:** {predicted_digit}")
    st.write(f"**Confidence:** {prediction[0][predicted_digit] * 100:.2f}%")

    # Show probabilities for all digits
    st.bar_chart(prediction[0])

# Prediction from drawn canvas
if canvas_result.image_data is not None:
    st.subheader("Prediction from Drawn Image:")
    # Convert canvas image to PIL Image
    drawn_image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    # Convert to grayscale
    drawn_image = drawn_image.convert('L')
    # Resize to 28x28
    drawn_image = drawn_image.resize((28, 28))
    # Convert to numpy array and normalize
    drawn_image_array = np.array(drawn_image) / 255.0
    # Reshape for model input
    drawn_image_array = drawn_image_array.reshape(1, 28, 28, 1)

    # Make prediction
    drawn_prediction = model.predict(drawn_image_array)
    drawn_predicted_digit = np.argmax(drawn_prediction)

    # Display prediction
    st.write(f"**Predicted Digit:** {drawn_predicted_digit}")
    st.write(f"**Confidence:** {drawn_prediction[0][drawn_predicted_digit] * 100:.2f}%")

    # Show probabilities for all digits
    st.bar_chart(drawn_prediction[0])
