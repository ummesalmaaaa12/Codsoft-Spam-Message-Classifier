import streamlit as st
import pickle
import numpy as np
import string
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load model & glove vectors
model = pickle.load(open("model.pkl", "rb"))
glove_vectors = pickle.load(open("glove_vectors.pkl", "rb"))

# Preprocessing setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Page setup
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")

# ------------------- CSS -------------------
css = """
<style>
body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #f1f2f6, #dfe4ea);
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, .label-text, .tagline {
    color: #000000 !important;
    font-weight: bold;
}

.stTextArea label {
    color: #000000 !important;
    font-size: 18px;
}

p, li {
    color: #000000;
    font-size: 16px;
}

/* Animated button */
.stButton > button {
    background-color: #2d7dd2;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    animation: pulse 2s infinite;
}

.stButton > button:hover {
    background-color: #1e5fa4;
    transform: scale(1.03);
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(45, 125, 210, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(45, 125, 210, 0); }
  100% { box-shadow: 0 0 0 0 rgba(45, 125, 210, 0); }
}

/* Result box */
.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.safe {
    border-left: 8px solid #00c851;
}
.spam {
    border-left: 8px solid #ff4444;
}

/* Footer */
.footer {
    text-align: center;
    color: #444;
    font-size: 15px;
    margin-top: 10px;
    padding-top: 15px;
    border-top: 1px solid #ccc;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ------------------- Dark Mode Toggle -------------------
dark_mode = st.toggle(" Enable Dark Mode")
if dark_mode:
    st.markdown("""
    <style>
    body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #121212, #232526);
    }
    h1, h2, h3, .label-text, .tagline, p, li {
        color: #ffffff !important;
    }
    .result-box {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .footer {
        color: #999999;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- Title & Description -------------------
st.markdown("<h1>SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='tagline'>AI-Powered SMS Filtering for Modern Communication</p>", unsafe_allow_html=True)
st.markdown("<h3 class='label-text'>Enter your message below:</h3>", unsafe_allow_html=True)

# ------------------- Input -------------------
message = st.text_area("", placeholder="Paste your SMS message here...", height=120)

# ------------------- Preprocessing -------------------
def preprocess_input(msg):
    msg = msg.lower()
    msg = ''.join([c for c in msg if c not in string.punctuation])
    words = msg.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def get_glove_vector(text):
    words = text.split()
    vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    return np.mean(vectors, axis=0).reshape(1, -1) if vectors else np.zeros((1, 100))

# ------------------- Prediction -------------------
if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        cleaned = preprocess_input(message)
        vector = get_glove_vector(cleaned)
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.markdown("""
            <div class="result-box spam">
                <h3 style="color:#d8000c;">Spam Message Detected</h3>
                <p>This message contains potential spam content. Be cautious before clicking any links or sharing personal info.</p>
                <ul>
                    <li>Common spam patterns identified</li>
                    <li>Suspicious language or urgency</li>
                    <li>Consider ignoring or blocking sender</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box safe">
                <h3 style="color:#007e33;">Not Spam !</h3>
                <p>This message appears <strong>safe</strong> and does not contain spam characteristics.</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown("""
<div class="footer">
Developed with precision by <strong>Ummesalma Rampurwala</strong> | © 2025
</div>
""", unsafe_allow_html=True)
