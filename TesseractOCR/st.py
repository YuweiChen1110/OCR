import streamlit as st
import pytesseract
from PIL import Image
import uuid
import os
import base64
import cv2
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Path to tessdata directory
TESSDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')

# Function to get all language models in tessdata directory
def get_language_models(tessdata_dir):
    lang_files = [f for f in os.listdir(tessdata_dir) if f.endswith('.traineddata')]
    languages = [os.path.splitext(f)[0] for f in lang_files]
    return languages

# Function to preprocess the image for better OCR results
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Function to run inference using multiple language models
def inference(img_path, langs):
    image = Image.open(img_path)
    result = pytesseract.image_to_string(image, lang='+'.join(langs))
    return result

# Function to create a download link
def create_download_link(file_path, file_name):
    with open(file_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'<a href="data:file/markdown;base64,{b64}" download="{file_name}">Download {file_name}</a>'
        return href

# Streamlit web app
st.title('TesseractOCR Web App V1.0')
st.write('This is a simple OCR web app using TesseractOCR with caching and improved code recognition.')

# Sidebar for inputs
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Get available language models
available_langs = get_language_models(TESSDATA_DIR)
langs = st.sidebar.multiselect('Select languages for OCR', available_langs, default=['eng'])

paragraph_file_name = st.sidebar.text_input("Enter the file name for recognized text download:", "recognized_text.md")

if uploaded_file is not None:
    with TemporaryDirectory() as temp_dir:
        bytes_data = uploaded_file.getvalue()
        unique_filename = os.path.join(temp_dir, str(uuid.uuid4()) + '.jpg')
        with open(unique_filename, 'wb') as f:
            f.write(bytes_data)
        st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)

        # Preprocess image
        binary_image = preprocess_image(unique_filename)
        processed_image_path = unique_filename.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_image_path, binary_image)

        if not langs:
            st.error("Please select at least one language for OCR.")
        else:
            # Run inference
            st.write("Recognizing text from image...")
            try:
                raw_results = inference(processed_image_path, langs)
                
                with st.expander("Raw OCR results"):
                    st.write(raw_results)
                
                if st.sidebar.button("Generate recognized text as markdown"):
                    with st.spinner('Processing...'):
                        with NamedTemporaryFile(delete=False, suffix=".md", dir=temp_dir) as tmp_file:
                            tmp_file.write(raw_results.encode('utf-8'))
                            tmp_file.flush()
                            tmp_file.seek(0)
                            href = create_download_link(tmp_file.name, paragraph_file_name)
                            st.sidebar.markdown(href, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
