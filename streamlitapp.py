import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_filter(image, filter_type):
    if filter_type == 'None':
        return image
    elif filter_type == 'Grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'Edge Detection':
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'Corner Detection':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        image[corners > 0.01 * corners.max()] = [0, 0, 255]
        return image

def main():
    st.title('Image Transformation App')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        transform_type = st.selectbox(
            'Choose a transformation',
            ('None', 'Grayscale', 'Edge Detection', 'Corner Detection')
        )

        if st.button('Transform'):
            transformed_image = apply_filter(image, transform_type)
            st.image(transformed_image, caption='Transformed Image', use_column_width=True)

if __name__ == "__main__":
    main()
