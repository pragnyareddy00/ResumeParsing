import streamlit as st
import os
import tempfile
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
import requests
from huggingface_hub import login

# Initialize NLTK and Spacy
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Hugging Face Token Setup
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("Hugging Face token not configured! Please set HF_TOKEN in secrets.")
    st.stop()
login(token=HF_TOKEN)

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# CSS styling
st.markdown("""
    <style>
    body { background-color: #f9f9f9; font-family: Arial, sans-serif; }
    .stButton>button, .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📄 Resume Analyzer")
st.markdown("Upload your resume and select a job role to get an ATS score and skill analysis.")

# Sidebar for inputs
with st.sidebar:
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, PNG, JPG)", 
                                   type=["pdf", "docx", "png", "jpg", "jpeg"])
    
    st.header("Select Job Role")
    role = st.selectbox(
        "Choose the job role to analyze against:",
        ["Data Science", "Database", "Designer", "Devops Engineer", "ETL", 
         "Developer", "Information Technology", "Python Developer", 
         "React Developer", "SAP Developer", "Testing"]
    )
    
    analyze_button = st.button("Analyze Resume")

# Text extraction functions
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

# NLP processing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(filtered_tokens))
    return [token.lemma_ for token in doc]

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

def get_ai_suggestions(missing_skills, role):
    """Generate suggestions using Hugging Face's API"""
    if not missing_skills:
        return "🎉 Great job! You have all the required skills for this role!"

    prompt = f"""
    Suggest 3-5 practical ways to develop these {role} skills: {', '.join(missing_skills)}.
    Focus on free resources and quick wins. Format as bullet points with emojis.
    """
    
    try:
        API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            return response.json()[0]['generated_text'].strip()
        else:
            return f"⚠️ API limit reached. Try:\n- Coursera (free courses)\n- freeCodeCamp\n- YouTube tutorials"

    except Exception as e:
        return f"🚧 Suggestions engine busy. Quick tips:\n1. GitHub practice projects\n2. Online coding challenges\n3. Professional networking"

# Main app logic
if analyze_button and uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            text = extract_text(tmp_path)
            if not text or len(text.strip()) < 10:
                raise ValueError("Extracted text is too short or empty")
            
            resume_tokens = preprocess_text(text)
            role_keywords = load_keywords(role)
            analysis_result = calculate_ats_score(resume_tokens, role_keywords)
            
            # Display results
            st.subheader(f"Analysis Results for {role} Role")
            score = analysis_result["score"]
            st.success(f"ATS Score: {score}/100")

            with st.expander("✅ Matching Skills", expanded=True):
                st.subheader("Required Skills Found")
                if analysis_result["matching_skills"]["required"]:
                    for skill in analysis_result["matching_skills"]["required"]:
                        st.markdown(f"- {skill.capitalize()}")
                else:
                    st.warning("No matching required skills found.")
                
                st.subheader("Optional Skills Found")
                if analysis_result["matching_skills"]["optional"]:
                    for skill in analysis_result["matching_skills"]["optional"]:
                        st.markdown(f"- {skill.capitalize()}")
                else:
                    st.info("No matching optional skills found.")

            with st.expander("❌ Missing Skills", expanded=True):
                st.subheader("Required Skills Missing")
                if analysis_result["missing_skills"]["required"]:
                    for skill in analysis_result["missing_skills"]["required"]:
                        st.markdown(f"- {skill.capitalize()}")
                else:
                    st.success("All required skills are present!")
                
                st.subheader("Optional Skills Missing")
                if analysis_result["missing_skills"]["optional"]:
                    for skill in analysis_result["missing_skills"]["optional"]:
                        st.markdown(f"- {skill.capitalize()}")
                else:
                    st.success("All optional skills are present!")

            with st.expander("📝 Resume Text Preview"):
                st.text_area("Extracted Text", value=text, height=300)

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
elif analyze_button and uploaded_file is None:
    st.warning("Please upload a resume file first.")

# AI Suggestions Section
with st.expander("💡 AI Suggestions for Missing Skills"):
    if uploaded_file and 'analysis_result' in locals():
        all_missing = analysis_result["missing_skills"]["required"] + analysis_result["missing_skills"]["optional"]
        if all_missing:
            with st.spinner("Generating AI suggestions..."):
                suggestions = get_ai_suggestions(all_missing, role)
                st.markdown(suggestions)
        else:
            st.success("No missing skills found! 🎉")
    else:
        st.info("Upload a resume and click 'Analyze' to get AI suggestions")

# Instructions section
with st.expander("ℹ️ How to Use This Tool"):
    st.markdown("""
    1. **Upload your resume** in PDF, DOCX, PNG, or JPG format  
    2. **Select the job role** you're applying for  
    3. Click **Analyze Resume** to get your ATS score and skill analysis
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Resume Analyzer Tool • Uses NLP to evaluate resume compatibility</p>
    </div>
""", unsafe_allow_html=True)
