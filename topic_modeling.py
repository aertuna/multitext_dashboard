import os, joblib, re
import pandas as pd
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

DATA_PATH  = 'data/covid_tweets.csv'
MODEL_PATH = 'models/lda.pkl'

def clean_text(text):
    txt = re.sub(r'http\\S+','', str(text))
    txt = re.sub(r'@\\w+','', txt)
    txt = re.sub(r'[^a-zA-Z]',' ', txt).lower()
    doc = nlp(txt)
    return [tok.lemma_ for tok in doc if tok.lemma_ not in STOPWORDS and len(tok.lemma_) > 2]

def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df = df[['OriginalTweet']].dropna().reset_index(drop=True)
    df['tokens'] = df['OriginalTweet'].apply(clean_text)
    return df

def train_or_load(num_topics=5, passes=3):
    os.makedirs('models', exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = load_and_prepare_data()
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(t) for t in df['tokens']]
    lda = models.LdaModel(corpus=corpus, id2word=dictionary,
                          num_topics=num_topics, passes=passes,
                          random_state=42)
    joblib.dump((lda, corpus, dictionary), MODEL_PATH)
    return lda, corpus, dictionary

def generate_topic_labels(lda_model, num_words=3):
    labels = []
    for topic_id, topic_words in lda_model.show_topics(num_topics=-1, formatted=False):
        top_keywords = [word for word, _ in topic_words[:num_words]]
        labels.append(" / ".join(top_keywords))
    return labels

def sample_topics(lda, corpus, dictionary, df, num_words=10, num_samples=2):
    topic_labels = generate_topic_labels(lda, num_words=3)

    print("\n\n=== Topic Labels ===")
    for i, label in enumerate(topic_labels):
        print(f"Topic {i}: {label}")

    print("\nSample tweets per topic:")
    for i, label in enumerate(topic_labels):
        print(f"\n--- {label} ---")
        docs = sorted([(idx, dict(lda.get_document_topics(bow)).get(i, 0))
                       for idx, bow in enumerate(corpus)],
                      key=lambda x: -x[1])[:num_samples]
        for idx, _ in docs:
            print(f"- {df['OriginalTweet'].iloc[idx]}")

    topic_counts = [0] * lda.num_topics
    for doc in corpus:
        topic_probs = lda.get_document_topics(doc)
        for topic_id, prob in topic_probs:
            if prob > 0.5:
                topic_counts[topic_id] += 1
                break

    used_topics = [(i, count, topic_labels[i]) for i, count in enumerate(topic_counts) if count > 0]
    used_topics.sort(key=lambda x: -x[1])

    if used_topics:
        topic_ids, counts, labels = zip(*used_topics)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(labels), y=list(counts), palette="Blues_d")
        plt.title("Topic Distribution (Filtered)")
        plt.ylabel("Document Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("Hiçbir doküman %50'den yüksek bir topic olasılığıyla eşleşmedi.")

def train_decision_tree_from_topics():
    df = load_and_prepare_data()
    
    # TF-IDF vektörleri
    texts = [" ".join(tokens) for tokens in df['tokens']]
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # LDA konuları (etiket olarak en baskın konuyu al)
    lda, corpus, dictionary = train_or_load()
    y = [max(lda.get_document_topics(doc), key=lambda x: x[1])[0] for doc in corpus]
    
    # Karar ağacı eğitimi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Doğruluk oranı
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nKarar Ağacı Doğruluk Oranı: {acc:.2f}")
    return acc

    if __name__ == "__main__":
        train_decision_tree_from_topics()
        df = load_and_prepare_data()
        lda, corpus, dictionary = train_or_load()
        sample_topics(lda, corpus, dictionary, df)
