import os, joblib
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

DATA_PATH = 'data/imdb.csv'
MODEL_PATH = 'models/imdb_sentiment.pkl'

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    return [w for w in text if w not in STOPWORDS]

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['review', 'sentiment']].dropna()
    df['tokens'] = df['review'].apply(clean_text)
    df['clean'] = df['tokens'].apply(lambda toks: ' '.join(toks))
    df['label'] = df['sentiment'].map({'positive':1, 'negative':0})
    return df

def train_or_load():
    os.makedirs('models', exist_ok=True)
    if os.path.exists(MODEL_PATH):
        model, vectorizer = joblib.load(MODEL_PATH)
    else:
        df = load_and_prepare_data()
        X = vectorizer = None
        vectorizer = TfidfVectorizer(max_features=2000)
        X_tfidf = vectorizer.fit_transform(df['clean'])
        model = LogisticRegression(max_iter=500)
        model.fit(X_tfidf, df['label'])
        joblib.dump((model, vectorizer), MODEL_PATH)
    return model, vectorizer

def predict_sample(model, vectorizer, text):
    toks = clean_text(text)
    vec = vectorizer.transform([' '.join(toks)])
    return 'Positive' if model.predict(vec)[0]==1 else 'Negative'

if __name__ == '__main__':
    model, vectorizer = train_or_load()
    sample = "I absolutely loved this movie, the story was great!"
    print(f"🔍 Prediction: {predict_sample(model, vectorizer, sample)}")
