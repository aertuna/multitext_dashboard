import os, joblib
from sentiment_imdb     import train_or_load as _imdb
from sentiment_ecommerce import train_or_load as _ecom
from sarcasm_detection  import train_or_load as _sar
from topic_modeling     import train_or_load as _lda

_models = {}

def init_models():
    if not _models:
        print("▶ Loading IMDB model...")
        _models['imdb'] = _imdb()
        print("▶ Loading E‑Com model...")
        _models['ecom'] = _ecom()
        print("▶ Loading Sarcasm model...")
        _models['sar']  = _sar()
        print("▶ Loading LDA model...")
        _models['topic']= _lda()
        print("✅ All models ready.")
    return _models

def predict_imdb_text(text):
    m,v = _models['imdb']
    from sentiment_imdb import predict_sample
    return predict_sample(m,v,text)

def predict_ecom_text(text):
    m,v = _models['ecom']
    from sentiment_ecommerce import predict_sample
    return predict_sample(m,v,text)

def predict_sar_text(text):
    m,v = _models['sar']
    from sarcasm_detection import predict_sample
    return predict_sample(m,v,text)

def get_topic_insights():
    lda, corpus, dict_ = _models['topic']
    from topic_modeling import sample_topics, pd
    df = pd.read_csv('data/covid_tweets.csv', encoding='latin-1')
    sample_topics(lda, corpus, dict_, df)
