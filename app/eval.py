
import os
import platform
import re
import shutil
import subprocess
from sys import exit
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from io import BytesIO
from PIL import Image  # Import Image from PIL

from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import pytesseract
import tempfile

# client_key_path= r"app/key.json"
current_os = platform.system().lower()
print(f"Current OS: {current_os}")


# Function to convert PDF to images and extract text
def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    return [extract_text_from_image(image_path) for image_path in images]


def extract_text_from_image(image):
    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
        # Save the image to the temporary file
        image.save(temp_file.name)

        # Open the saved image file
        with Image.open(temp_file.name) as img:
            # Convert PIL image to grayscale
            img = img.convert('L')

            # Use pytesseract to extract text from the image
            text = pytesseract.image_to_string(img)

    return text



def separate_answers(texts):
    separated_answers = {}

    last_answer = None
    for i, text in enumerate(texts):
        answers = re.split(r'(?=Answer \d+:)', text)

        for answer in answers:
            match = re.match(r'Answer \d+:', answer)
            if match:
                answer_number = match.group(0)
                text = answer[len(answer_number):].strip()
                last_answer = f"{i + 1}_{answer_number}"
                separated_answers[last_answer] = text
            elif last_answer is not None:
                separated_answers[last_answer] += f" {answer.strip()}"

    return separated_answers


def calculate_similarity(teacher_answers, student_answers):
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    scores = {}

    for student_key, student_answer in student_answers.items():
        teacher_answer = teacher_answers.get(student_key, "")

        student_tokens = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
        teacher_tokens = tokenizer(teacher_answer, return_tensors="pt", padding=True, truncation=True)
        student_embedding = model(**student_tokens).last_hidden_state.mean(dim=1).detach().numpy()
        teacher_embedding = model(**teacher_tokens).last_hidden_state.mean(dim=1).detach().numpy()

        similarity_score = cosine_similarity(student_embedding, teacher_embedding)[0][0]
        scores[student_key] = similarity_score

    return scores
