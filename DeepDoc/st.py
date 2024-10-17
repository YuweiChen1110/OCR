import streamlit as st
import os
import subprocess
import tempfile
import shutil
from zipfile import ZipFile

# Streamlit page title
st.title('DeepDoc Web App V1.0')

# File upload
uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

# Check if a file has been uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_file_path = tmp_file.name

    # Get the directory of the input file
    input_dir = os.path.dirname(input_file_path)

    # Set the output folder to 'recognized_content'
    output_folder = os.path.join(input_dir, 'recognized_content')

    # Display the uploaded file name and the fixed output folder
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.write(f"Fixed output folder: {output_folder}")

    # Button to trigger the processing script
    if st.button("Start Processing"):
        # If the output folder does not exist, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Call the trans.py script
        result = subprocess.run(
            ['python', 'trans.py', '--input_file', input_file_path, '--output_folder', output_folder],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Display the script output
        st.write("Script output:")
        st.text(result.stdout)

        # Display errors (if any)
        if result.stderr:
            st.write("Error output:")
            st.text(result.stderr)

        # Zip the recognized_content folder
        zip_file_path = os.path.join(input_dir, 'recognized_content.zip')
        with ZipFile(zip_file_path, 'w') as zip_file:
            for foldername, subfolders, filenames in os.walk(output_folder):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_file.write(file_path, os.path.relpath(file_path, output_folder))

        # Provide a download button for the user to download the zip file
        with open(zip_file_path, 'rb') as f:
            st.download_button(
                label="Download Processed Results",
                data=f,
                file_name='recognized_content.zip',
                mime='application/zip'
            )

# Prompt if no file is uploaded
else:
    st.write("Please upload a PDF file")
