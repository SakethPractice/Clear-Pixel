from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from src.convolution import process_image


st.set_page_config(page_title="Clear-Pixel", page_icon="🖼️", layout="wide")


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_download_bytes(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode the processed image.")
    return buffer.tobytes()


st.title("Clear-Pixel")
st.subheader("Apply matrix convolution for image blurring and sharpening")

st.markdown(
    """
    ### What the dashboard can do
    - Upload an image
    - Select operation: blur or sharpen
    - Choose kernel size
    - Show original image
    - Show processed image
    - Download the output image
    """,
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

mode = st.selectbox("Select operation", options=["blur", "sharpen"])
kernel_size = st.slider("Choose kernel size", min_value=3, max_value=15, step=2, value=3)

if uploaded_file is not None:
    try:
        original_image = decode_uploaded_image(uploaded_file.getvalue())
        processed_image, kernel = process_image(original_image, mode, kernel_size)

        original_column, processed_column = st.columns(2)

        with original_column:
            st.markdown("#### Original image")
            st.image(to_rgb(original_image), width="stretch")

        with processed_column:
            st.markdown("#### Processed image")
            st.image(to_rgb(processed_image), width="stretch")

        st.markdown("#### Kernel used")
        st.code(np.array2string(kernel, precision=3), language="text")

        st.download_button(
            label="Download output image",
            data=to_download_bytes(processed_image),
            file_name=f"{mode}_output.png",
            mime="image/png",
        )
    except ValueError as error:
        st.error(str(error))
else:
    st.info("Upload an image to start applying convolution.")
