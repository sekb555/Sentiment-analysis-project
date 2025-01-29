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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk import punkt
stop_words = set(stopwords.words('english'))



BATCH_SIZE = 64
MAX_EPOCHS = 10
CHUNK_SIZE = 170000


def preprocess_text(text):
    # remove users tags and any URLs that mat be present and replase then with USER and URL respectively
    text = text.astype(str).str.replace(r'@\w+', 'USER', regex=True)
    text = text.astype(str).str.replace(r'http\S+', 'URL', regex=True)
    text = text.astype(str).str.replace(r'\W', '')
    text = text.astype(str).str.strip()
    text = text.astype(str).apply(html.unescape)
    return text


def process_date(date):
    date = date.astype(str).str.split()
    date = pd.DataFrame(date.tolist(), columns=[
                        'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
    date = date.drop(columns=['Timezone', 'Time'])
    return date


def word_freq(words):
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)
    print(common_words)

    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(' '.join(words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('word cloud of the most common words')
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
    "data/training.1600000.processed.noemoticon.csv", header=None, encoding="ISO-8859-1")
df.columns = ["Polarity", "ID", "Date", "Flag", "User", "Tweet"]

# assign text and sentiment to variables
twts = df["Tweet"]
sentiments = df["Polarity"]

dates = df["Date"]
dates_split = dates.astype(str).str.split()
dates = pd.DataFrame(dates_split.tolist(), columns=[
                     'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
dates = dates.drop(columns=['Timezone', 'Time', 'Day'])

twts = preprocess_text(twts)

sentiments = (sentiments == 4).astype(int)  # 0 for negative, 1 for positive

words = twts.astype(str).str.split().explode()
word_list = []
for word in words:
    if word not in stop_words:
        word_list.append(word)


df["DateOnly"] = pd.to_datetime(
    dates['Month'] + " " + dates['Date'] + " " + dates['Year'])
df["sentiments"] = sentiments
grouped = df.groupby("DateOnly")['sentiments'].value_counts().unstack().fillna(0)
sentiment_over_time(grouped)



