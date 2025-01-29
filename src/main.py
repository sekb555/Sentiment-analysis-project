import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords

import html

BATCH_SIZE = 64
MAX_EPOCHS = 10
CHUNK_SIZE = 170000

# read the training data and assign the columns names
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv", header=None, encoding="ISO-8859-1")
df.columns = ["Polarity", "ID", "Date", "Flag", "User", "Tweet"]

# assign text and sentiment to variables
twts = df["Tweet"]
sentiments = df["Polarity"]


def preprocess_text(text):
    # remove users tags and any URLs that mat be present and replase then with USER and URL respectively
    text = text.astype(str).str.replace(r'@\w+', 'USER', regex=True)
    text = text.astype(str).str.replace(r'http\S+', 'URL', regex=True)
    text = text.astype(str).str.replace(r'\W', '')
    text = text.astype(str).str.strip()
    text = text.astype(str).apply(html.unescape)
    return text


twts = preprocess_text(twts)

words = twts.astype(str).str.split().explode()
unique_words = set(words)

sentiments = (sentiments == 4).astype(int)  # 0 for negative, 1 for positive
# sentiments.info()

train_twts, test_twts, train_labels, test_labels = train_test_split(
    twts, sentiments, test_size=0.15, random_state=42)


print(words.head(19))
