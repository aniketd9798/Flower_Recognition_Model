import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. PAGE SETUP
st.set_page_config(page_title="AI Flower Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Flower Recognition ")
st.markdown("Upload a picture of a flower, and the custom-built Artificial Intelligence will guess what it is!")

# 2. LOAD THE AI BRAIN
# We use @st.cache_resource so Streamlit only loads the heavy AI model once, making the app much faster.
@st.cache_resource
def load_flower_model():
    # Make sure this matches the exact name you saved in train_brain.py
    return tf.keras.models.load_model("custom_flower_model.h5")

try:
    model = load_flower_model()
except OSError:
    st.error("Model not found! Please run `train_brain.py` first to generate your custom_flower_model.h5 file.")
    st.stop()

# Keras flow_from_directory sorts folders alphabetically by default.
# If your folders are named exactly like the Kaggle dataset, this is the exact order:
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# 3. UPLOAD AND DISPLAY IMAGE
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image the user uploaded
    image = Image.open(uploaded_file)
    # Convert image to RGB just in case the user uploads a transparent PNG (RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("🧠 The AI is analyzing the image...")

    # 4. PREPROCESS THE IMAGE (Must match exactly how we trained it!)
    # Resize to 150x150 pixels
    img_resized = image.resize((150, 150))
    # Convert the image to a NumPy array of numbers
    img_array = np.array(img_resized)
    # Scale the colors from 0-255 down to 0-1 (just like rescaling in ImageDataGenerator)
    img_array = img_array / 255.0
    # Add a "batch" dimension because the AI expects a list of images, not just one.
    img_array = np.expand_dims(img_array, axis=0)

    # 5. MAKE THE PREDICTION
    predictions = model.predict(img_array)[0] # Get the first (and only) result
    
    # Find the highest probability
    highest_confidence_index = np.argmax(predictions)
    predicted_flower = CLASS_NAMES[highest_confidence_index]
    confidence_score = predictions[highest_confidence_index] * 100

    # 6. DISPLAY THE RESULTS
    st.markdown("---")
    if confidence_score > 70:
        st.success(f"### 🎯 Prediction: {predicted_flower}")
    else:
        st.warning(f"### 🤔 Best Guess: {predicted_flower}")
        st.caption("The AI isn't highly confident about this one. Try another angle or better lighting!")
        
    st.info(f"**Confidence Score:** {confidence_score:.2f}%")

    # Optional: Show a bar chart of all probabilities so your professors can see the AI's "thought process"
    st.markdown("##### The AI's Thought Process:")
    chart_data = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    st.bar_chart(chart_data)