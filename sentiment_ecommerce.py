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

DATA_PATH = 'data/ecommerce.csv'
MODEL_PATH = 'models/ecom_sentiment.pkl'

def clean_text(text):
    text = re.sub(r'<.*?>',' ', str(text))
    text = re.sub(r'[^a-zA-Z]',' ', text).lower().split()
    return [w for w in text if w not in STOPWORDS]

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['Review Text','Rating']].dropna()
    df['tokens'] = df['Review Text'].apply(clean_text)
    df['clean']  = df['tokens'].apply(lambda t:' '.join(t))
    df['label']  = df['Rating'].apply(lambda x: 1 if x>3 else 0)
    return df

def train_or_load():
    os.makedirs('models', exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = load_and_prepare_data()
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df['clean'])
    model = LogisticRegression(max_iter=500)
    model.fit(X, df['label'])
    joblib.dump((model,vectorizer), MODEL_PATH)
    return model, vectorizer

def predict_sample(model, vectorizer, text):
    toks = clean_text(text)
    vec = vectorizer.transform([' '.join(toks)])
    return 'Positive' if model.predict(vec)[0]==1 else 'Negative'

if __name__ == '__main__':
    m, v = train_or_load()
    print(predict_sample(m, v, "This dress fits perfectly and the material is amazing!"))
