import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def load_data():
    fake1 = pd.read_csv('data/politifact_fake.csv')
    real1 = pd.read_csv('data/politifact_real.csv')
    fake2 = pd.read_csv('data/gossipcop_fake.csv')
    real2 = pd.read_csv('data/gossipcop_real.csv')

    # Labels
    fake1['label'] = 0
    fake2['label'] = 0
    real1['label'] = 1
    real2['label'] = 1

    # Combine all data
    data = pd.concat([fake1, fake2, real1, real2])

    # 🔥 IMPORTANT FIX: Use ONLY 'title'
    data['text'] = data['title'].apply(clean_text)

    return data[['text', 'label']]
