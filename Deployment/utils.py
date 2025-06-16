import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array

# Hardcoded labels based on your training folder
def gen_labels():
    return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Image preprocessing
def preprocess(image):
    image = image.resize((150, 150), Image.Resampling.LANCZOS)  # Resize to model input shape
    image = img_to_array(image)
    image = image / 255.0
    return image

# Model architecture — must match your training setup
def model_arc():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(6, activation='softmax')  # 6 classes
    ])
    return model
