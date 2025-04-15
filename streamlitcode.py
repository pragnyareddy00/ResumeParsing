import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Set paths
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
RESUME_PATH = "C:/Users/Pragnya Reddy/Downloads/resumes_project/Resumes"

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def extract_text_from_image(img_path):
    # Read image using OpenCV
    img = cv2.imread(img_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply OCR
    text = pytesseract.image_to_string(thresh)
    
    return text

def load_and_preprocess_data(resume_path):
    resumes = []
    labels = []  # You can modify this to include actual labels if available
    
    for filename in os.listdir(resume_path):
        if filename.endswith('.png'):
            img_path = os.path.join(resume_path, filename)
            text = extract_text_from_image(img_path)
            processed_text = preprocess_text(text)
            resumes.append(processed_text)
            labels.append(1)  # Dummy label - replace with actual labels if available
    
    return resumes, labels

# LSTM Model
def create_lstm_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Keyword matching function
def calculate_keyword_score(resume_text, job_description_keywords):
    vectorizer = TfidfVectorizer()
    
    # Combine resume and keywords for vectorization
    documents = [resume_text, ' '.join(job_description_keywords)]
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Scale to 0-100
    return round(similarity * 100, 2)

# Streamlit App
def main():
    st.title("Resume ATS Scoring System")
    st.write("Upload your resume in PNG format to get an ATS score and improvement suggestions")
    
    # Load or train model
    model_path = "resume_lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        model = load_model(model_path)
        tokenizer = pd.read_pickle(tokenizer_path)
    else:
        # Train model (this would normally be done separately)
        st.warning("Model not found. Training a new model...")
        resumes, labels = load_and_preprocess_data(RESUME_PATH)
        
        # Tokenize text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(resumes)
        vocab_size = len(tokenizer.word_index) + 1
        
        # Convert text to sequences
        sequences = tokenizer.texts_to_sequences(resumes)
        max_length = max([len(seq) for seq in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Train LSTM model
        model = create_lstm_model(vocab_size, max_length)
        model.fit(padded_sequences, np.array(labels), epochs=10, batch_size=32, validation_split=0.2)
        
        # Save model and tokenizer
        model.save(model_path)
        pd.to_pickle(tokenizer, tokenizer_path)
    
    # Job description keywords (customize this for different job roles)
    job_description_keywords = [
        'python', 'machine learning', 'deep learning', 'data analysis',
        'sql', 'tensorflow', 'pytorch', 'natural language processing',
        'computer vision', 'statistics', 'data visualization', 'pandas',
        'numpy', 'scikit-learn', 'aws', 'docker', 'git', 'problem solving',
        'communication', 'teamwork', 'project management'
    ]
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a resume (PNG format)", type="png")
    
    if uploaded_file is not None:
        # Save uploaded file
        with open("temp_resume.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text
        resume_text = extract_text_from_image("temp_resume.png")
        processed_text = preprocess_text(resume_text)
        
        # Display extracted text
        with st.expander("View Extracted Text"):
            st.text(resume_text)
        
        # Calculate scores
        keyword_score = calculate_keyword_score(processed_text, job_description_keywords)
        
        # Prepare text for LSTM
        sequences = tokenizer.texts_to_sequences([processed_text])
        max_length = model.input_shape[1]
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        
        # Get LSTM prediction
        lstm_score = model.predict(padded_sequences)[0][0] * 100
        
        # Combine scores (weighted average)
        final_score = (keyword_score * 0.6 + lstm_score * 0.4)
        
        # Display scores
        st.subheader("ATS Score")
        st.progress(int(final_score))
        st.write(f"**Overall ATS Score:** {final_score:.1f}/100")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Keyword Matching Score:** {keyword_score:.1f}/100")
        with col2:
            st.write(f"**Content Quality Score:** {lstm_score:.1f}/100")
        
        # Generate suggestions
        st.subheader("Improvement Suggestions")
        
        # Check for missing keywords
        missing_keywords = [kw for kw in job_description_keywords if kw not in processed_text]
        if missing_keywords:
            st.write("**Add these relevant keywords:**")
            st.write(", ".join(missing_keywords[:10]))  # Show first 10 missing keywords
        
        # Check resume length
        word_count = len(processed_text.split())
        if word_count < 200:
            st.write("**Consider adding more details:** Your resume seems quite short.")
        elif word_count > 800:
            st.write("**Consider making it more concise:** Your resume might be too long.")
        
        # Check for sections
        required_sections = ['experience', 'education', 'skills', 'projects']
        missing_sections = [sec for sec in required_sections if sec not in resume_text.lower()]
        if missing_sections:
            st.write("**Add these missing sections:**")
            st.write(", ".join(missing_sections))
        
        # Formatting tips
        st.write("**Formatting Tips:**")
        st.write("- Use clear section headings")
        st.write("- Use bullet points for readability")
        st.write("- Include measurable achievements")
        st.write("- Keep consistent formatting throughout")
        
        # Visualize keyword matches
        st.subheader("Keyword Analysis")
        matched_keywords = [kw for kw in job_description_keywords if kw in processed_text]
        unmatched_keywords = [kw for kw in job_description_keywords if kw not in processed_text]
        
        fig, ax = plt.subplots()
        sns.barplot(x=['Matched', 'Unmatched'], 
                    y=[len(matched_keywords), len(unmatched_keywords)],
                    palette=['green', 'red'])
        plt.title("Keyword Matching")
        st.pyplot(fig)

if __name__ == "__main__":
    main()