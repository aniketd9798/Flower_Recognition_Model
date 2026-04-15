# Flower_Recognition_Model

##  Project Overview
This project is an end-to-end Deep Learning pipeline that classifies images of flowers into five distinct categories: **Daisy, Dandelion, Rose, Sunflower, and Tulip**. 

Unlike projects that rely on pre-trained models (Transfer Learning), this system features a **Custom Convolutional Neural Network (CNN)** built entirely from scratch using TensorFlow and Keras. The project also includes an interactive web dashboard built with Streamlit for real-time, user-friendly predictions.

##  Key Features
* **Custom Architecture:** A multi-block CNN designed from scratch with Convolutional, MaxPooling, and Dropout layers.
* **Automated Data Cleaning:** Built-in Python scripting to automatically scan and purge corrupted or hidden system files.
* **Data Augmentation:** Utilizes `ImageDataGenerator` to artificially expand the training data via rotation, zoom, and flipping.
* **Interactive Web Dashboard:** A sleek Streamlit interface allowing users to upload images and view confidence scores.
* **CLI Testing Utility:** A dedicated script to test individual image predictions directly from the terminal.

##  Technology Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** Pillow (PIL), NumPy
* **Web Deployment:** Streamlit
* **Language:** Python 3.10+

##  Project Structure

Flower_CNN_Project/
│
├── dataset/                  # Contains subfolders for each flower class
├── app.py                    # Streamlit Web Dashboard
├── train_brain.py            # CNN Architecture and Training script
├── test_prediction.py        # Command-line testing utility
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation

## Installation & Setup

**1. Clone the repository**

git clone : (https://github.com/aniketd9798/Flower_Recognition_Model)

**2. Create a Virtual Environment**

python -m venv flower_env
flower_env\Scripts\activate   # On Windows

**3. Install Dependencies**

pip install -r requirements.txt

##  How to Run

**Step 1: Train the Model**
Run the training script to generate the `custom_flower_model.h5` file.

python train_brain.py

**Step 2: Test via Command Line**
Place an image named `test_flower.jpg` in the root directory and run:

python test_prediction.py

**Step 3: Launch the Web Dashboard**

streamlit run app.py

##  Future Scope
* Implement MobileNetV2 for comparison against custom architecture.
* Expand dataset to include more species.
* Deploy the dashboard to Streamlit Cloud.
