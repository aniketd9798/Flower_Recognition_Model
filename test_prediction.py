import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ==========================================
# 1. SETUP
# ==========================================
MODEL_PATH = "custom_flower_model.h5"
TEST_IMAGE_PATH = "test_flower.jpg"  # You will need to put a test image in your folder
CLASS_NAMES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Check if the model exists before trying to load it
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Could not find {MODEL_PATH}. Please wait for train_brain.py to finish!")
    exit()

# Check if the user remembered to add a test image
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"[ERROR] Could not find {TEST_IMAGE_PATH}.")
    print("-> Please download a picture of a flower from Google.")
    print("-> Place it in this project folder.")
    print("-> Rename the picture to exactly 'test_flower.jpg' and run this script again.")
    exit()

# ==========================================
# 2. LOAD MODEL & IMAGE
# ==========================================
print("[INFO] Loading the AI brain...")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"[INFO] Processing {TEST_IMAGE_PATH}...")
# Resize the image to match the exact size the CNN expects (150x150)
img = image.load_img(TEST_IMAGE_PATH, target_size=(150, 150))

# Convert the image pixels into a NumPy array of numbers
img_array = image.img_to_array(img)

# Scale the pixel brightness from 0-255 down to 0-1 (just like we did during training)
img_array = img_array / 255.0

# Add a "batch" dimension because the AI expects a list of images, even if it's just one
img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# 3. MAKE PREDICTION
# ==========================================
print("[INFO] Asking the AI for a prediction...\n")
# Get the raw probabilities for all 5 flower types
predictions = model.predict(img_array, verbose=0)[0]

# Find the highest probability
highest_confidence_index = np.argmax(predictions)
predicted_flower = CLASS_NAMES[highest_confidence_index]
confidence_score = predictions[highest_confidence_index] * 100

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================
print("="*40)
print(f"🎯 FINAL PREDICTION: {predicted_flower}")
print(f"📊 CONFIDENCE SCORE: {confidence_score:.2f}%")
print("="*40)

print("\n[Detailed Breakdown of AI's Thought Process]")
for i, flower in enumerate(CLASS_NAMES):
    print(f" -> {flower}: {predictions[i]*100:.2f}%")