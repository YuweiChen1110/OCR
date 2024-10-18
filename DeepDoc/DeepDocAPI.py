import os
import subprocess
import shutil
import tempfile
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from zipfile import ZipFile

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/process-pdf/', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            input_file_path = os.path.join(temp_dir, filename)
            file.save(input_file_path)

            output_folder = os.path.join(temp_dir, 'recognized_content')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            result = subprocess.run(
                ['python', 'ragflow/deepdoc/vision/t_ocr.py', '--inputs', input_file_path, '--output_dir', output_folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return jsonify({"error": f"Script error: {result.stderr}"}), 500

            zip_file_path = os.path.join(temp_dir, 'recognized_content.zip')
            with ZipFile(zip_file_path, 'w') as zip_file:
                for foldername, subfolders, filenames in os.walk(output_folder):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        zip_file.write(file_path, os.path.relpath(file_path, output_folder))

            return send_file(zip_file_path, as_attachment=True, attachment_filename='recognized_content.zip')

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

        finally:
            shutil.rmtree(temp_dir)

    else:
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
