import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def load_data():
    df = pd.read_csv("/Users/sekb/Desktop/processed_data.csv",
                 encoding="ISO-8859-1")
    return df

# separates the words in the text data
def extract_words(text):
    words = text.astype(str).str.split().explode()
    return words


# plot the most common phrases in the text data
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


# plot the most common words in the text data
def word_freq(words):
    word_freq = Counter(words)
    common_words = word_freq.most_common(20)

    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*common_words))
    plt.xlabel('Words')
    plt.ylabel('Amount')
    plt.title('Word Frequency')
    plt.show()


# plot the sentiment distribution
def sentiment_distribution(sentiments):
    plt.figure(figsize=(6, 4))
    sentiments.value_counts().plot(kind='bar')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.show()


# plot the sentiment distribution over time
def sentiment_over_time(S_D):
    plt.figure(figsize=(10, 5))
    S_D.plot(kind='line', stacked=True)
    plt.title('Sentiment Distribution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.show()


df = load_data()
df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)

while (1):
    EDA = input("Enter the EDA function you want to run: ")
    EDA = EDA.lower()
    if EDA == "word frequency":
        word_freq(extract_words(df['Processed_Tweets']))
    elif EDA == "common phrases":
        common_phrases(df['Processed_Tweets'])
    elif EDA == "sentiment distribution":
        sentiment_distribution(df['Polarity'])
    elif EDA == "sentiment distribution over time":
        sentiment_over_time(df.groupby("DateOnly")[
                            'Polarity'].value_counts().unstack().fillna(0))
    elif EDA == "exit" or EDA == "":
        break
    else:
        print("sorry that function does not exist try again")
