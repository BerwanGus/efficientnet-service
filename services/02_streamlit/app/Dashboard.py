import os
import streamlit as st
import numpy as np
from PIL import Image
from io import StringIO

st.title(":blue[Upload image]", help='images can be uploaded through this button \
          and the EfficientNet model will predict their respective classes.')

uploaded_files = st.file_uploader("Choose a file", type=['jpeg', 'png'], accept_multiple_files=True)
if uploaded_files is not None:
    for file in uploaded_files:
        st.image(file)

