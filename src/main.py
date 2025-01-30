import html
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import punkt
stop_words = stopwords.words('english')


BATCH_SIZE = 64
MAX_EPOCHS = 10
CHUNK_SIZE = 170000
VOCAB_SIZE = 922770
MAX_LEN = 32


def preprocess_text(text):
    text = text.astype(str).str.replace(
        r'@\w+', '', regex=True)  # remove user tags
    text = text.astype(str).str.replace(
        r'http\S+', '', regex=True)  # remove URLs
    text = text.astype(str).str.replace(
        r'[^a-zA-Z0-9 ]', '', regex=True)  # remove special characters
    # remove leading and trailing whitespaces
    text = text.astype(str).str.strip()
    text = text.apply(html.unescape)  # remove html
    return text


def word_seperator(text):
    words = text.astype(str).str.split().explode()
    word_list = []
    for word in words:
        if word not in stop_words:
            word_list.append(word)
    return word_list


def process_date(date):
    date = date.astype(str).str.split()
    date = pd.DataFrame(date.tolist(), columns=[
                        'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
    date = date.drop(columns=['Timezone', 'Time'])
    return date


def common_phrases(texts):
    phrases = []
    for text in texts:
        text = text.lower()
        phrase = text.split('.')
        phrases.extend(phrase)

    phrase_freq = Counter(phrases)
    common_phrases = phrase_freq.most_common(3)

    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*common_phrases))
    plt.xlabel('Phrases')
    plt.ylabel('Amount')
    plt.title('Common Phrases')
    plt.show()


def word_freq(words):
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)

    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*common_words))
    plt.xlabel('Words')
    plt.ylabel('Amount')
    plt.title('Word Frequency')
    plt.show()


def sentiment_distribution(sentiments):
    plt.figure(figsize=(6, 4))
    sentiments.value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()


def sentiment_over_time(S_D):
    plt.figure(figsize=(10, 5))
    # S_D.plot(kind='line')
    S_D.plot(kind='line', stacked=True)
    plt.title('Sentiment Distribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.show()


# read the training data and assign the columns names
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv", header=None, encoding="ISO-8859-1", nrows=1000)
df.columns = ["Polarity", "ID", "Date", "Flag", "User", "Tweet"]

# assign text and sentiment to variables
twts = df["Tweet"]
sentiments = df["Polarity"]
sentiments = (sentiments == 4).astype(int)  # 0 for negative, 1 for positive

# preprocess the input data
twts = preprocess_text(twts)
# convert the tweet into individual words and remove stop words
word_list = word_seperator(twts)

# split the date column and remove unnecessary columns
dates = df["Date"]
dates_split = dates.astype(str).str.split()
dates = pd.DataFrame(dates_split.tolist(), columns=[
                     'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
dates = dates.drop(columns=['Timezone', 'Time', 'Day'])

df["DateOnly"] = pd.to_datetime(
    dates['Month'] + " " + dates['Date'] + " " + dates['Year'])
df["sentiments"] = sentiments
grouped = df.groupby("DateOnly")[
    'sentiments'].value_counts().unstack().fillna(0)


train_twts, test_twts, train_sentiments, test_sentiments = train_test_split(
    twts, sentiments, test_size=0.15, random_state=42)
