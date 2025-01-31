import pandas as pd
import numpy as np
from collections import Counter
import re
import pickle

from preprocess_data import preprocess_text, remove_stopwords
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

lr_model = pickle.load(open('data/logistic_regression_model.sav', 'rb'))

while True:
    text = input("Enter a tweet: ")
    if text == "exit" or text == '':
        break
    text = preprocess_text(text)
    text = remove_stopwords(text)
    print(text)
    prediction = lr_model.predict([text])
    print(prediction)

    if prediction == 0:
        print("Negative sentiment")
    else:
        print("Positive sentiment")



