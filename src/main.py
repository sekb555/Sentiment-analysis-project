import pandas as pd
import numpy as np
from collections import Counter
import re
import joblib

from preprocess_data import PreprocessData as ppd
import matplotlib.pyplot as plt


import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

lr_model = joblib.load(open('data/logistic_regression_model.sav', 'rb'))

while True:
    text = input("Enter a tweet: ")
    if text == "exit" or text == '':
        break
    text = ppd.preprocess_text(text)
    text = ppd.remove_stopwords(text)
    print(text)
    prediction = lr_model.predict([text])
    print(prediction)

    if prediction == 0:
        print("Negative sentiment")
    else:
        print("Positive sentiment")
