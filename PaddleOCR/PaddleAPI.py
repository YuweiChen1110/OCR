from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import shutil
import uuid
import os
import cv2
import urllib.request
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

app = Flask(__name__)

# Clean the cache to avoid corrupted files issue
cache_dir = os.path.expanduser('~/.paddleocr')

def remove_readonly(func, path, _):
    os.chmod(path, 0o777)
    func(path)

if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir, onerror=remove_readonly)
    except PermissionError as e:
        print(f"PermissionError encountered while deleting cache directory: {e}")

# Model download links
model_urls = {
    'det': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_ppocr_server_v2.0_det_infer.tar',
    'rec': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/en_ppocr_server_v2.0_rec_infer.tar',
    'cls': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
}

# Download and extract models
def download_and_extract(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename}")

    print(f"Extracting {filename}...")
    with tarfile.open(filepath, 'r') as tar:
        tar.extractall(path=output_dir)
    print(f"Extracted {filename}")

# Check and download models
def check_and_download_models():
    model_dirs = {
        'det': 'ch_ppstructure_mobile_v2.0_SLANet_infer',
        'rec': 'models/rec/en_PP-OCRv4_rec_infer'
    }
    for key, model_dir in model_dirs.items():
        if not os.path.exists(model_dir):
            download_and_extract(model_urls[key], model_dir)

# Perform model check and download
check_and_download_models()

# Global variable for caching the OCR model
ocr_model_cache = None

# Function to load the OCR model with caching
def load_ocr_model():
    global ocr_model_cache
    if ocr_model_cache is not None:
        return ocr_model_cache
    
    det_model_dir = 'models/det/ch_ppstructure_mobile_v2.0_SLANet_infer'
    rec_model_dir = 'models/rec/en_PP-OCRv4_rec_infer'
    
    ocr_model_cache = PaddleOCR(
        use_angle_cls=True, 
        lang='en', 
        det_model_dir=det_model_dir, 
        rec_model_dir=rec_model_dir
    )
    return ocr_model_cache

# Load the OCR model initially
load_ocr_model()

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
    
    # Extract text from OCR result and merge it into a single paragraph
    merged_text = ""
    for line in result[0]:  # result[0] contains the OCR text lines
        merged_text += line[1][0] + " "  # line[1][0] is the recognized text

    merged_text = merged_text.strip()  # Remove trailing spaces
    return merged_text

@app.route('/ocr', methods=['POST'])
def ocr_service():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    with TemporaryDirectory() as temp_dir:
        unique_filename = os.path.join(temp_dir, str(uuid.uuid4()) + '.jpg')
        image_file.save(unique_filename)

        # Preprocess the image
        binary_image = preprocess_image(unique_filename)
        processed_image_path = unique_filename.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_image_path, binary_image)

        # Run OCR
        try:
            paragraph_text = inference(processed_image_path)
            return jsonify({'recognized_text': paragraph_text})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
