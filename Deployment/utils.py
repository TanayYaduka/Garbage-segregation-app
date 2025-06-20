from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

def gen_labels():
    return ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess(image):
    image = image.resize((300, 300), Image.Resampling.LANCZOS)  # ✅ FIXED HERE
    image = np.array(image) / 255.0
    return image


def model_arc():
    model = Sequential()

    # Convolution blocks
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', input_shape=(300,300,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Classification layers
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    # Enable OneDNN optimizations
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    return model
