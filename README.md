# Codsoft-Spam-Message-Classifier
This project is a real-time SMS Spam Classifier Web App developed as part of my Machine Learning Internship at CodSoft. It combines Natural Language Processing (NLP) techniques with Machine Learning to detect whether an incoming SMS is spam or legitimate (ham).

Overview
The aim of this project is to provide users with an interactive tool that analyzes and classifies SMS messages using a trained machine learning model. It also includes a visually appealing frontend using Streamlit, built for ease of use and a better user experience.

Features
Classifies SMS messages as Spam or Ham

Real-time prediction with text preprocessing pipeline

Custom word embedding using GloVe vectors

Clean and professional UI built using Streamlit

Message-level security guidance for users

Developed with a balance of backend logic and frontend presentation

Tech Stack
Python

Scikit-learn for machine learning

NLTK for NLP preprocessing

GloVe Word Embeddings for vector representation

NumPy for array and matrix operations

Streamlit for frontend web app interface

Pickle for model and vector serialization

Model Details
Model Used: Support Vector Machine (SVM)

Vectorization: GloVe (Global Vectors for Word Representation)

Text Preprocessing:

Lowercasing

Removing punctuation

Removing stop words

Word stemming using Porter Stemmer

Installation & Setup
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
Install required libraries

nginx
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

arduino
Copy
Edit
streamlit run app.py
Files
app.py - Main web application code

model.pkl - Trained SVM model

glove_vectors.pkl - Preprocessed GloVe embeddings dictionary

README.md - Project overview and documentation

What I Learned
Through this project, I gained hands-on experience in:

Implementing a complete machine learning pipeline for text classification

Working with GloVe word embeddings and vectorization

Integrating NLP techniques with model training

Deploying machine learning models using Streamlit

Designing a user-oriented frontend interface with professional styling

Author
Ummesalma Rampurwala
Machine Learning Intern | CodSoft

