import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import gdown
import os

from utils import gen_labels, preprocess, model_arc  # make sure utils.py contains these functions

# Generate labels
labels = gen_labels()

# HTML and UI Design
html_temp = '''
  <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: -50px">
    <div style="display: flex; flex-direction: row; align-items: center; justify-content: center;">
      <center><h1 style="color: #000; font-size: 50px;"><span style="color: #0e7d73">Smart </span>Garbage</h1></center>
      <img src="https://cdn-icons-png.flaticon.com/128/1345/1345823.png" style="width: 0px;">
    </div>
    <div style="margin-top: -20px">
      <img src="https://i.postimg.cc/W3Lx45QB/Waste-management-pana.png" style="width: 400px;">
    </div>  
  </div>
'''
st.markdown(html_temp, unsafe_allow_html=True)

st.markdown('''
  <div>
    <center><h3 style="color: #008080; margin-top: -20px">Check the type here </h3></center>
  </div>
''', unsafe_allow_html=True)

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload Option
opt = st.selectbox("How do you want to upload the image for classification?", 
                   ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None

if opt == 'Upload image from device':
    file = st.file_uploader('Select an image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':
    try:
        img_url = st.text_input('Enter the Image Address')
        if img_url:
            image = Image.open(urllib.request.urlopen(img_url))
    except:
        if st.button('Submit'):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

# If image is uploaded, show preview
if image is not None:
    st.image(image, width=300, caption='Uploaded Image')

    if st.button('Predict'):
        st.info("Processing...")

        # Preprocess image
        img = preprocess(image)

        # Download model from Google Drive if not already present
        file_id = '19UR26IpbztBfWM_uVMlE6rv240sssr9D'
        output_path = 'modelnew.h5'
        if not os.path.exists(output_path):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

        # Load model
        model = model_arc()
        model.load_weights(output_path)

        # Make prediction
        prediction = model.predict(img[np.newaxis, ...])
        result = labels[np.argmax(prediction[0], axis=-1)]

        st.success(f'Hey! The uploaded image has been classified as **"{result} product"**.')

