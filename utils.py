import re
import json
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def extract_text(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.endswith('.pdf'):
            try:
                return extract_text_from_pdf(file_path)
            except Exception as pdf_error:
                print(f"PDF extraction failed, trying OCR: {pdf_error}")
                return extract_text_from_image(file_path)
                
        elif file_path.endswith('.docx'):
            return extract_text_from_docx(file_path)
            
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return extract_text_from_image(file_path)
            
        else:
            raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
            
    except Exception as e:
        print(f"Error in extract_text: {str(e)}")
        raise

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return lemmatized_tokens

def load_keywords(role):
    role_files = {
        "Data Science": "data_science_profile.json",
        "Database": "database_profile.json",
        "Designer": "designer_profile.json",
        "Devops Engineer": "devops_engineer_profile.json",
        "ETL": "etl_developer_profile.json",
        "Information Technology": "information_technology_profile.json",
        "Python Developer": "python_developer_profile.json",
        "React Developer": "react_developer_profile.json",
        "SAP Developer": "sap_developer_profile.json",
        "Testing": "testing_profile.json"
    }
    
    file_path = os.path.join("role_keywords", role_files.get(role))
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_ats_score(resume_tokens, role_keywords):
    required_skills = role_keywords.get("required_skills", [])
    optional_skills = role_keywords.get("optional_skills", [])
    
    matching_required = [skill for skill in required_skills if skill in resume_tokens]
    matching_optional = [skill for skill in optional_skills if skill in resume_tokens]
    
    missing_required = [skill for skill in required_skills if skill not in resume_tokens]
    missing_optional = [skill for skill in optional_skills if skill not in resume_tokens]
    
    # Calculate score (70% weight to required skills, 30% to optional)
    req_score = (len(matching_required) / len(required_skills)) * 70 if required_skills else 0
    opt_score = (len(matching_optional) / len(optional_skills)) * 30 if optional_skills else 0
    total_score = req_score + opt_score
    
    return {
        "score": round(total_score, 2),
        "matching_skills": {
            "required": matching_required,
            "optional": matching_optional
        },
        "missing_skills": {
            "required": missing_required,
            "optional": missing_optional
        }
    }

import requests

def get_ai_suggestions(missing_skills, role):
    """
    Generates AI-based suggestions for the given missing skills and job role using Hugging Face's Inference API.
    """
    if not missing_skills:
        return "Great job! You have all the required skills for this role."

    prompt = (
        f"As a career advisor, suggest how to acquire the following missing skills for a {role} role: "
        f"{', '.join(missing_skills)}. Provide practical steps or resources."
    )

    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/starchat-alpha"
    headers = {"Authorization": "Bearer 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"}  # Replace with your actual Hugging Face API key

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=30)
        response.raise_for_status()
        generated_text = response.json()[0]['generated_text']
        return generated_text.strip()
    except Exception as e:
        return f"Error fetching AI suggestions: {e}"
