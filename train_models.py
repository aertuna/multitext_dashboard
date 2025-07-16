import os
import joblib
import pandas as pd
import re
import json
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from gensim import corpora, models

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')

DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

STOPWORDS = set(stopwords.words('english'))

def clean_text_generic(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' ', str(text))
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return [w for w in text if w not in STOPWORDS]

def train_imdb():
    from sentiment_imdb import load_and_prepare_data
    df = load_and_prepare_data()
    model, vec = sentiment_imdb.train_model(df)
    joblib.dump((model, vec), f"{MODEL_DIR}/imdb_sentiment.pkl")

def train_ecom():
    from sentiment_ecommerce import load_and_prepare_data, train_model
    df = load_and_prepare_data()
    model, vec = train_model(df)
    joblib.dump((model, vec), f"{MODEL_DIR}/ecom_sentiment.pkl")

def train_sarcasm():
    from sarcasm_detection import load_and_prepare_data, train_model
    df = load_and_prepare_data()
    model, vec = train_model(df)
    joblib.dump((model, vec), f"{MODEL_DIR}/sarcasm.pkl")

def train_lda():
    from topic_modeling import load_and_prepare_data, train_lda
    df = load_and_prepare_data()
    lda_model, corpus, dictionary = train_lda(df, num_topics=5, passes=3)
    joblib.dump((lda_model, corpus, dictionary), f"{MODEL_DIR}/lda.pkl")

if __name__ == '__main__':
    train_imdb()
    train_ecom()
    train_sarcasm()
    train_lda()
    print("✅ All models trained and saved to models/ directory")
