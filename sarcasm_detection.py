import os, joblib, json, re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

DATA_PATH  = 'data/sarcasm.json'
MODEL_PATH = 'models/sarcasm.pkl'

def clean_text(text):
    text = re.sub(r'https?://\\S+',' ', text)
    text = re.sub(r'<.*?>',' ', text)
    text = re.sub(r'[^a-zA-Z]',' ', text).lower().split()
    return [w for w in text if w not in STOPWORDS]

def load_and_prepare_data():
    heads, labs = [], []
    with open(DATA_PATH) as f:
        for line in f:
            j = json.loads(line)
            heads.append(j['headline'])
            labs.append(j['is_sarcastic'])
    df = pd.DataFrame({'headline':heads,'label':labs})
    df['tokens'] = df['headline'].apply(clean_text)
    df['clean']  = df['tokens'].apply(lambda t:' '.join(t))
    return df

def train_or_load():
    os.makedirs('models', exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = load_and_prepare_data()
    vect = TfidfVectorizer(max_features=2000)
    X = vect.fit_transform(df['clean'])
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    model.fit(X, df['label'])
    joblib.dump((model,vect), MODEL_PATH)
    return model, vect

def predict_sample(model, vect, text):
    toks = clean_text(text)
    vec  = vect.transform([' '.join(toks)])
    return 'Sarcastic' if model.predict(vec)[0]==1 else 'Not Sarcastic'

if __name__ == '__main__':
    m, v = train_or_load()
    print(predict_sample(m, v, "10 Amazing Facts About The Moon You Probably Didn't Know"))
