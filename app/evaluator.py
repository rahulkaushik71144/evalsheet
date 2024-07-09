import os
import platform
import re
import shutil
import subprocess
from sys import exit
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from io import BytesIO


from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

client_key_path= r"app/key.json"
current_os = platform.system().lower()
print(f"Current OS: {current_os}")



# Function to convert PDF to images and extract text
def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    return [extract_text_from_image(image) for image in images]

def extract_text_from_image(image_content):
    # Set up the Google Cloud Vision API client with the service account key
    client = vision_v1.ImageAnnotatorClient.from_service_account_file(client_key_path)

    # Convert PIL Image to bytes
    with BytesIO() as byte_io:
        image_content.save(byte_io, format='PNG')
        byte_value = byte_io.getvalue()

    # Create Image message with bytes content
    input_image = types.Image(content=byte_value)
    response = client.text_detection(image=input_image)


    texts = response.full_text_annotation
    return texts.text.replace('\n', ' ')


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