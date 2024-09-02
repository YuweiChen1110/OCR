from flask import Flask, request, jsonify, send_file
import pytesseract
from PIL import Image
import uuid
import os
import base64
import cv2
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory

app = Flask(__name__)

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

# Endpoint to list available language models
@app.route('/languages', methods=['GET'])
def list_languages():
    languages = get_language_models(TESSDATA_DIR)
    return jsonify(languages)

# Endpoint to perform OCR
@app.route('/ocr', methods=['POST'])
def ocr_service():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    langs = request.form.get('langs', 'eng').split(',')

    image_file = request.files['image']

    # Save the image temporarily
    with TemporaryDirectory() as temp_dir:
        unique_filename = os.path.join(temp_dir, str(uuid.uuid4()) + '.jpg')
        image_file.save(unique_filename)

        # Preprocess the image
        binary_image = preprocess_image(unique_filename)
        processed_image_path = unique_filename.replace('.jpg', '_processed.jpg')
        cv2.imwrite(processed_image_path, binary_image)

        # Run OCR
        try:
            result = inference(processed_image_path, langs)
            return jsonify({'recognized_text': result})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
