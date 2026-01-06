import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    tokens = text.split()
    filtered = [word for word in tokens if word not in STOPWORDS]
    return " ".join(filtered)


def preprocess_resume(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
