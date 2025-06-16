import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import os
from utils import *

# === DOWNLOAD MODEL FROM GOOGLE DRIVE ===
model_url = "https://drive.google.com/uc?id=19UR26IpbztBfWM_uVMlE6rv240sssr9D"
model_path = "modelnew.h5"

if not os.path.exists(model_path):
    with st.spinner("Downloading model weights..."):
        urllib.request.urlretrieve(model_url, model_path)
        st.success("Model downloaded successfully!")

# === LABELS ===
labels = gen_labels()

# === PAGE UI ===
html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style = "display: flex; flex-direction: row; align-items: center; justify-content: center;">
     <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
    <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
    <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

html_temp = '''
    <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here </h3></center>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)

# === IMAGE UPLOAD OPTIONS ===
opt = st.selectbox("How do you want to upload the image for classification?\n", 
                   ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None
if opt == 'Upload image from device':
    file = st.file_uploader('Select an image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
    img_url = st.text_input('Enter the Image Address')
    if img_url:
        try:
            image = Image.open(urllib.request.urlopen(img_url))
        except:
            st.error("Please enter a valid image URL.")

# === PREDICTION ===
if image is not None:
    st.image(image, width=300, caption='Uploaded Image')
    if st.button('Predict'):
        img = preprocess(image)

        model = model_arc()
        model.load_weights(model_path)

        prediction = model.predict(img[np.newaxis, ...])
        predicted_class = labels[np.argmax(prediction[0], axis=-1)]

        st.info(f'Hey! The uploaded image has been classified as **"{predicted_class} product"**.')
