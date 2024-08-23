import streamlit as st
from paddleocr import PaddleOCR
import uuid
import os
import base64
import cv2
from tempfile import NamedTemporaryFile, TemporaryDirectory

# Global variable for caching the OCR model
ocr_model_cache = None

# Function to load the OCR model with caching
@st.cache_resource
def load_ocr_model():
    global ocr_model_cache
    if ocr_model_cache is not None:
        return ocr_model_cache
    
    ocr_model_cache = PaddleOCR(
        use_angle_cls=True, 
        lang='en'
    )
    return ocr_model_cache

# Function to clear the OCR model cache
def clear_cache():
    global ocr_model_cache
    ocr_model_cache = None

# Function to preprocess the image for better OCR results
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
    return binary_image

# Function to run inference using the cached OCR model
def inference(img_path):
    ocr = load_ocr_model()
    image = cv2.imread(img_path)  # Read the processed image
    result = ocr.ocr(image, cls=True)  # Pass image array directly
    return result

# Function to extract and sort text by spatial order
def extract_sorted_text(ocr_result):
    text_info = [(line[0], line[1][0], line[1][1]) for result in ocr_result for line in result]
    text_info_sorted = sorted(text_info, key=lambda x: (x[0][0][1], x[0][0][0]))
    return text_info_sorted

# Function to merge text lines into paragraphs based on their vertical spacing
def merge_lines_to_paragraphs(text_info, line_spacing_threshold=10):
    paragraphs = []
    current_paragraph = []
    last_y = None
    
    for line_info in text_info:
        box, text, confidence = line_info
        top_left_y = box[0][1]

        if last_y is not None and top_left_y - last_y > line_spacing_threshold:
            paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

        current_paragraph.append(text)
        last_y = top_left_y

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))
    
    return paragraphs

# Function to convert paragraphs to markdown
def convert_paragraphs_to_markdown(paragraphs, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for paragraph in paragraphs:
            file.write(f"{paragraph}\n\n")

# Function to create a download link
def create_download_link(file_path, file_name):
    with open(file_path, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'<a href="data:file/markdown;base64,{b64}" download="{file_name}">Download {file_name}</a>'
        return href

# Streamlit web app
st.title('Paddle OCR Web App V1.0')
st.write('This is a simple OCR web app using PaddleOCR with caching and improved code recognition.')

# Cache management buttons
st.sidebar.header("Cache Management")
if st.sidebar.button("Clear Cache"):
    clear_cache()
    st.sidebar.success("Cache cleared.")

if st.sidebar.button("Reload Model"):
    clear_cache()
    load_ocr_model()
    st.sidebar.success("Model reloaded.")

# Load the OCR model initially
load_ocr_model()

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
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

        # Run inference
        st.write("Recognizing text from image...")
        try:
            raw_results = inference(processed_image_path)
            sorted_text_info = extract_sorted_text(raw_results)
            paragraphs = merge_lines_to_paragraphs(sorted_text_info)
            
            with st.expander("Raw OCR results"):
                for line in sorted_text_info:
                    st.write(f"Detected text: {line[1]} (Confidence score: {line[2]})")

            st.write("Paragraphs:")
            for paragraph in paragraphs:
                st.write(paragraph)

            # User input for file names in sidebar
            paragraph_file_name = st.sidebar.text_input("Enter the file name for recognized text download:", "recognized_text.md")
            
            # Create download link in sidebar
            if st.sidebar.button("Generate recognized text as markdown"):
                with st.spinner('Processing...'):
                    with NamedTemporaryFile(delete=False, suffix=".md", dir=temp_dir) as tmp_file:
                        convert_paragraphs_to_markdown(paragraphs, tmp_file.name)
                        tmp_file.flush()
                        tmp_file.seek(0)
                        href = create_download_link(tmp_file.name, paragraph_file_name)
                        st.sidebar.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(unique_filename):
                os.remove(unique_filename)
            if os.path.exists(processed_image_path):
                os.remove(processed_image_path)