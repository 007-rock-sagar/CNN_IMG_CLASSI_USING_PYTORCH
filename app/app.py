import streamlit as st
from PIL import Image
import numpy as np
from utils import load_model, preprocess_image
from streamlit_drawable_canvas import st_canvas

# Load your trained model (adjust path if needed)
model = load_model("../mnist_cnn.pth")

st.title("🖼️ CNN Image Classifier (PyTorch)")
st.write("Upload an image or draw a digit on the writing pad.")

# --- Option 1: Upload an image ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=300)

    input_tensor = preprocess_image(image)
    output = model(input_tensor)
    _, predicted = output.max(1)

    st.success(f"Prediction from uploaded image: {predicted.item()}")

# --- Option 2: Writing pad ---
st.write("Or draw a digit below:")

canvas_result = st_canvas(
    fill_color="black",          # background color
    stroke_width=10,             # pen thickness
    stroke_color="white",        # pen color
    background_color="black",    # match MNIST background
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:
    # Convert canvas to grayscale PIL image
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    st.image(img, caption="Your Drawing", width=200)

    input_tensor = preprocess_image(img)
    output = model(input_tensor)
    _, predicted = output.max(1)

    st.success(f"Prediction from writing pad: {predicted.item()}")
