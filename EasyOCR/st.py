import streamlit as st
import easyocr
from PIL import Image
import uuid
import os
import base64
import cv2
import numpy as np
from io import BytesIO
from tempfile import NamedTemporaryFile

# List of supported languages (you can update this list based on your needs)
SUPPORTED_LANGUAGES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'az': 'Azerbaijani', 'bg': 'Bulgarian',
    'cs': 'Czech', 'da': 'Danish', 'de': 'German', 'en': 'English',
    'es': 'Spanish', 'fa': 'Persian', 'fr': 'French', 'ga': 'Irish',
    'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian',
    'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean',
    'mn': 'Mongolian', 'ms': 'Malay', 'nl': 'Dutch', 'no': 'Norwegian',
    'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian',
    'sl': 'Slovenian', 'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'Swedish',
    'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'zh-cn': 'Chinese Simplified', 'zh-tw': 'Chinese Traditional'
}

# Function to preprocess the image for better OCR results
def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert('L')
    np_image = np.array(image)
    _, binary_image = cv2.threshold(np_image, 150, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Function to run inference using EasyOCR and preserve paragraph formatting
def inference_with_formatting(img_array, langs):
    reader = easyocr.Reader(langs)
    results = reader.readtext(img_array)
    
    # Initialize variables for paragraph processing
    paragraphs = []
    current_paragraph = []
    last_bottom = None

    for bbox, text, prob in results:
        # bbox is a list of four points (each a tuple of x, y coordinates)
        bottom = max(bbox, key=lambda x: x[1])[1]  # Get the maximum y value (bottom)

        if last_bottom is not None:
            # Check vertical space to determine if a new paragraph is needed
            if abs(bottom - last_bottom) > 15:  # You can adjust the threshold as needed
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []

        current_paragraph.append(text)
        last_bottom = bottom

    # Add the last paragraph
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    
    # Join all paragraphs with line breaks
    formatted_text = "\n\n".join(paragraphs)
    return formatted_text

# Function to create a download link
def create_download_link(file_path, file_name):
    with open(file_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'<a href="data:file/markdown;base64,{b64}" download="{file_name}">Download {file_name}</a>'
        return href

# Streamlit web app
st.title('EasyOCR Web App V1.0')
st.write('This is an improved version of EasyOCR Web App that preserves paragraph formatting.')

# Sidebar - Settings and actions
st.sidebar.header('Settings')

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Language selection in the sidebar
available_langs = list(SUPPORTED_LANGUAGES.keys())
langs = st.sidebar.multiselect('Select languages for OCR', available_langs, default=['en'])

# Input for file name and button to generate file in the sidebar
paragraph_file_name = st.sidebar.text_input("Enter the file name for recognized text download:", "recognized_text.md")

if uploaded_file is not None and langs:
    # Read the uploaded file
    bytes_data = uploaded_file.getvalue()

    # Display the uploaded image
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)

    # Preprocess image
    binary_image = preprocess_image(bytes_data)

    # Run inference with loading spinner
    st.write("Recognizing text from image...")
    with st.spinner('Processing...'):
        try:
            raw_results = inference_with_formatting(binary_image, langs)

            with st.expander("Formatted OCR results"):
                st.write(raw_results)

            if st.sidebar.button("Generate recognized text as markdown"):
                with st.spinner('Generating file...'):
                    with NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
                        tmp_file.write(raw_results.encode('utf-8'))
                        tmp_file.flush()
                        tmp_file.seek(0)
                        href = create_download_link(tmp_file.name, paragraph_file_name)
                        st.sidebar.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
elif not langs:
    st.error("Please select at least one language for OCR.")
