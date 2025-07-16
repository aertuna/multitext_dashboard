import os
import re
import pandas as pd
import nltk
import spacy
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from gensim import corpora, models
from functools import lru_cache

nltk.download('stopwords', quiet=True)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

DATA_DIR = 'data'

import pickle

@lru_cache(maxsize=None)
def get_lda(dataset):
    model_path = os.path.join(DATA_DIR, f'{dataset}_lda.pkl')
    dict_path = os.path.join(DATA_DIR, f'{dataset}_dict.pkl')
    corpus_path = os.path.join(DATA_DIR, f'{dataset}_corpus.pkl')
    texts_path = os.path.join(DATA_DIR, f'{dataset}_texts.pkl')

    if all(os.path.exists(p) for p in [model_path, dict_path, corpus_path, texts_path]):
        with open(model_path, 'rb') as f:
            lda_model = pickle.load(f)
        with open(dict_path, 'rb') as f:
            dictionary = pickle.load(f)
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)
        return lda_model, corpus, dictionary, texts

    if dataset == 'covid':
        df = pd.read_csv(os.path.join(DATA_DIR,'covid_tweets.csv'), encoding='latin-1')
        texts = df['OriginalTweet'].dropna().astype(str).tolist()
    elif dataset == 'imdb':
        df = pd.read_csv(os.path.join(DATA_DIR,'imdb.csv'))
        texts = df['review'].dropna().astype(str).tolist()
    elif dataset == 'ecom':
        df = pd.read_csv(os.path.join(DATA_DIR,'ecommerce.csv'))
        texts = df['Review Text'].dropna().astype(str).tolist()
    else:
        import json
        texts = []
        for line in open(os.path.join(DATA_DIR,'sarcasm.json'), encoding='utf-8'):
            texts.append(json.loads(line)['headline'])

    tokenized = []
    for t in texts:
        cleaned = re.sub(r'[^a-zA-Z]', ' ', t).lower()
        doc = nlp(cleaned)
        tokenized.append([tok.lemma_ for tok in doc if tok.lemma_ not in STOPWORDS and len(tok.lemma_)>2])

    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary,
                                num_topics=4, passes=5, random_state=42)

    with open(model_path, 'wb') as f:
        pickle.dump(lda_model, f)
    with open(dict_path, 'wb') as f:
        pickle.dump(dictionary, f)
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
    with open(texts_path, 'wb') as f:
        pickle.dump(texts, f)

    return lda_model, corpus, dictionary, texts

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Multitext Mining Dashboard'

app.layout = html.Div(className='container', children=[
    html.H1('Multitext Mining Dashboard', className='header'),
    dcc.Tabs(id='tabs', value='tab-imdb', children=[
        dcc.Tab(label='IMDB Sentiment', value='tab-imdb'),
        dcc.Tab(label='E-Commerce Sentiment', value='tab-ecom'),
        dcc.Tab(label='Sarcasm Detection', value='tab-sar'),
        dcc.Tab(label='Topic Modeling', value='tab-topic'),
    ]),
    html.Div(id='tabs-content', className='content')
])

@app.callback(Output('tabs-content', 'children'), Input('tabs','value'))
def render_tab(tab):
    def load_examples(dataset, col, num=2):
        path = os.path.join(DATA_DIR, dataset)
        if dataset.endswith('.csv'):
            df = pd.read_csv(path)
            return df[col].dropna().astype(str).head(num).tolist()
        elif dataset.endswith('.json'):
            import json
            texts = []
            with open(path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num: break
                    texts.append(json.loads(line)['headline'])
            return texts
        return []

    if tab == 'tab-imdb':
        examples = load_examples('imdb.csv', 'review')
        return html.Div([
            html.H3('IMDB Sentiment Analysis'),
            html.Div([html.B('Examples:'), html.Ul([html.Li(e) for e in examples])]),
            dcc.Textarea(id='imdb-input', placeholder='Enter review...', className='textarea'),
            html.Button('Analyze', id='imdb-btn', n_clicks=0, className='btn'),
            html.Div(id='imdb-output', className='output')
        ])
    if tab == 'tab-ecom':
        examples = load_examples('ecommerce.csv', 'Review Text')
        return html.Div([
            html.H3('E-Commerce Sentiment Analysis'),
            html.Div([html.B('Examples:'), html.Ul([html.Li(e) for e in examples])]),
            dcc.Textarea(id='ecom-input', placeholder='Enter review...', className='textarea'),
            html.Button('Analyze', id='ecom-btn', n_clicks=0, className='btn'),
            html.Div(id='ecom-output', className='output')
        ])
    if tab == 'tab-sar':
        examples = load_examples('sarcasm.json', '')
        return html.Div([
            html.H3('Sarcasm Detection'),
            html.Div([html.B('Examples:'), html.Ul([html.Li(e) for e in examples])]),
            dcc.Textarea(id='sar-input', placeholder='Enter headline...', className='textarea'),
            html.Button('Detect', id='sar-btn', n_clicks=0, className='btn'),
            html.Div(id='sar-output', className='output')
        ])
    if tab == 'tab-topic':
        return html.Div([
            html.H3('Topic Modeling'),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[
                    {'label':'COVID Tweets','value':'covid'},
                    {'label':'IMDB Reviews','value':'imdb'},
                    {'label':'E-Commerce Reviews','value':'ecom'},
                    {'label':'News Headlines','value':'sar'}
                ],
                value='covid', className='dropdown'
            ),
            html.Button('Run Topic Modeling', id='topic-btn', n_clicks=0, className='btn'),
            dcc.Loading(html.Div(id='topic-output', className='topic-container'), type='circle')
        ])
    return html.Div()

@app.callback(Output('imdb-output','children'), Input('imdb-btn','n_clicks'), State('imdb-input','value'))
def cb_imdb(n, text):
    if n and text:
        from sentiment_imdb import train_or_load, predict_sample
        model, vec = train_or_load()
        return f'Result: {predict_sample(model, vec, text)}'
    return ''

@app.callback(Output('ecom-output','children'), Input('ecom-btn','n_clicks'), State('ecom-input','value'))
def cb_ecom(n, text):
    if n and text:
        from sentiment_ecommerce import train_or_load, predict_sample
        model, vec = train_or_load()
        return f'Result: {predict_sample(model, vec, text)}'
    return ''

@app.callback(Output('sar-output','children'), Input('sar-btn','n_clicks'), State('sar-input','value'))
def cb_sar(n, text):
    if n and text:
        from sarcasm_detection import train_or_load, predict_sample
        model, vec = train_or_load()
        return f'Result: {predict_sample(model, vec, text)}'
    return ''

@app.callback(
    Output('topic-output','children'),
    Input('topic-btn','n_clicks'),
    State('dataset-dropdown','value')
)
def cb_topic(n, dataset):
    if not n:
        return html.Div('No data yet.', className='info')
    lda_model, corpus, dictionary, texts = get_lda(dataset)
    num_topics = lda_model.num_topics
    blocks = []
    for tid in range(num_topics):
        terms = lda_model.get_topic_terms(tid, topn=4)
        words = [dictionary[id] for id, _ in terms]
        weights = [weight for _, weight in terms]

        sorted_pairs = sorted(zip(weights, words))
        wts, wds = zip(*sorted_pairs)
        fig = go.Figure(go.Bar(x=list(wts), y=list(wds), orientation='h'))
        fig.update_layout(margin=dict(l=100, r=20, t=30, b=20), height=240, title=f'Topic {tid}')
        chart = dcc.Graph(figure=fig, className='chart')
        doc_scores = [(i, dict(lda_model.get_document_topics(corpus[i])).get(tid,0)) for i in range(len(corpus))]
        top_idx = sorted(doc_scores, key=lambda x:-x[1])[:2]
        examples = html.Ul([html.Li(texts[i][:100] + '...') for i,_ in top_idx], className='tweet-list')
        blocks.append(html.Div([chart, examples], className='topic-block'))
    return blocks

if __name__ == '__main__':
    app.run(debug=True)

