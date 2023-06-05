import os
import streamlit as st
import numpy as np
from PIL import Image
from io import StringIO

st.title(":blue[Upload image]", help='images can be uploaded through this button \
          and the EfficientNet model will predict their respective classes.')

uploaded_files = st.file_uploader("Choose a file", type=['jpeg', 'png'], accept_multiple_files=True)

if st.button("Generate predictions"):
    if len(uploaded_files) > 0:
        for file in uploaded_files:
            img = Image.open(file)
            img.resize((32, 32))
    else:
        st.write("No images provided.")


