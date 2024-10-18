import os
import subprocess
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from zipfile import ZipFile

# Initialize FastAPI app
app = FastAPI()

# Define the route for uploading the PDF file and processing it
@app.post("/deepdoc-api/")
async def deepdoc_api(file: UploadFile = File(...)):
    # Ensure the uploaded file is a PDF
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    try:
        # Save the uploaded PDF to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            input_file_path = tmp_file.name

        # Define the output folder in the same directory as input file
        input_dir = os.path.dirname(input_file_path)
        output_folder = os.path.join(input_dir, 'recognized_content')

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Call the trans.py script
        result = subprocess.run(
            ['python', 'ragflow/deepdoc/vision/t_ocr.py', '--inputs', input_file_path, '--output_dir', output_folder],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check for errors in the script execution
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script error: {result.stderr}")

        # Create a ZIP file from the output folder
        zip_file_path = os.path.join(input_dir, 'recognized_content.zip')
        with ZipFile(zip_file_path, 'w') as zip_file:
            for foldername, subfolders, filenames in os.walk(output_folder):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_file.write(file_path, os.path.relpath(file_path, output_folder))

        # Return the ZIP file as a downloadable response
        return FileResponse(zip_file_path, filename='recognized_content.zip')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        # Cleanup: remove temporary files and directories
        if os.path.exists(input_file_path):
            os.remove(input_file_path)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
