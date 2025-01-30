import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
import re

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def word_seperator(text):
    words = text.astype(str).str.split().explode()
    # word_list = []
    # for word in words:
    #     if word not in stop_words:
    #         word_list.append(word)
    return words


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
    
df = pd.read_csv("data/processed_data.csv", encoding="ISO-8859-1")
df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)

word_freq(word_seperator(df['Processed_Tweets']))


