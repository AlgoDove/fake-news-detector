import streamlit as st
import joblib
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag, word_tokenize
import os
import sys

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- 1. Load Assets (Model and Vectorizer) ---
# NOTE: This loads the files from your local models/ folder
try:
    model_path = 'models/champion_lr_model.pkl'
    vectorizer_path = 'models/fitted_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Error: Model files not found. Check if they are in the 'models/' folder.")
        st.stop() 

    lr_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
except Exception as e:
    st.error(f"Failed to load model or vectorizer. Make sure joblib is installed. Error: {e}")
    st.stop()

# --- 2. Preprocessing Logic ---
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    required_resources = ['punkt', 'wordnet', 'omw-1.4', 
                          'averaged_perceptron_tagger', 'stopwords']
    for resource in required_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_and_vectorize(text, vectorizer):
    text = text.lower().strip()
    words = word_tokenize(text)
    
    cleaned_words = [lemmatizer.lemmatize(
                        word.translate(str.maketrans('', '', string.punctuation)),
                        get_wordnet_pos(pos)
                     )
                     for word, pos in pos_tag(words)
                     if word not in stop_words and word.strip() != '']
                     
    processed_text = ' '.join(cleaned_words)
    vectorized_text = vectorizer.transform([processed_text])
    
    return vectorized_text

# --- 3. Streamlit UI and Prediction Logic ---

st.title("📰 Real or Fake? News Article Classifier")
st.markdown("---")
st.markdown(
    """
    <p style='font-size: 1.1em; color: #555;'>
    This app uses a Logistic Regression model (97% F1 Score) and TF-IDF features to predict the veracity of a news article.
    </p>
    """, unsafe_allow_html=True
)

article_title = st.text_input(
    "1. Enter the Article Title (Optional but Recommended)", 
    "Top tech companies report record-breaking Q3 earnings.", 
    max_chars=200
)

article_text = st.text_area(
    "2. Paste the Full Article Text Here",
    "In a surprise announcement, major technology corporations, including AlphaCorp and BetaTech, have announced earnings significantly higher than analyst expectations. The surge is attributed to increased cloud service adoption and strong holiday pre-orders.",
    height=300
)

predict_button = st.button("Analyze Article", type="primary")

if predict_button and (article_title or article_text):
    input_text = f"{article_title} {article_text}"
    
    with st.spinner('Analyzing content...'):
        X_input = clean_and_vectorize(input_text, tfidf_vectorizer)
        
        prediction = lr_model.predict(X_input)[0]
        prediction_proba = lr_model.predict_proba(X_input)[0][prediction]
        
    st.markdown("## Analysis Result:")

    if prediction == 1:
        st.success(f"**REAL NEWS** (Confidence: {prediction_proba*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"**FAKE NEWS** (Confidence: {prediction_proba*100:.2f}%)")

elif predict_button and not (article_title or article_text):
    st.warning("Please paste some text into the box to analyze.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF (Max 5000, 1-2 Gram)")