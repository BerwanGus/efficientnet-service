import os
import requests
import streamlit as st

st.title(":blue[Upload image]", help='images can be uploaded through this button \
          and the EfficientNet model will predict their respective classes.')

uploaded_files = st.file_uploader("Choose a file", type=['jpeg', 'png'], accept_multiple_files=True)

if st.button("Generate predictions"):
    if len(uploaded_files) > 0:
        files = {file.name:file.read() for file in uploaded_files}
        response = requests.get(os.getenv('API_URL') + '/predict', files=files)
        st.write(response.status_code, response.json())
    else:
        st.write("No images provided.")
