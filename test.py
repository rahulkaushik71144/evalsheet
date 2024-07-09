import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import os

def pdf_to_text(pdf_path):
    # Create a temporary directory to store the images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert PDF to a list of PIL images
        images = convert_from_path(pdf_path, output_folder=temp_dir)

        # Initialize an empty list to store text from all images
        all_text = []

        # Iterate through each image and extract text
        for i, image in enumerate(images):
            # Convert PIL image to grayscale
            image = image.convert('L')

            # Use pytesseract to extract text from the image
            text = pytesseract.image_to_string(image)

            # Append the extracted text to the list
            all_text.append(text)

    # Join all the extracted text from images
    result_text = '\n'.join(all_text)
    return result_text

# Example usage:
pdf_path = "teacher_answers.pdf"
result_text = pdf_to_text(pdf_path)
print(result_text)
