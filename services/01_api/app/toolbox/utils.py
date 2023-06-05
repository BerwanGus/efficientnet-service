import numpy as np
from PIL import Image
from io import BytesIO

def resize_image(file: BytesIO):
    img = Image.open(file)
    img = np.array()