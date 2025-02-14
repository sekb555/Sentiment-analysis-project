import pandas as pd
import numpy as np
from collections import Counter
import re
import joblib

from preprocess_data import PreprocessData
import matplotlib.pyplot as plt

ppd = PreprocessData()

import nltk
stop_words = nltk.download('stopwords')

# load the logistic regression model
lr_model = joblib.load(open('data/logistic_regression_model.sav', 'rb'))

# This code is for testing the sentiment of a sentence
while True:
    text = input("Enter some text: ")
    if text == "exit":
        break
    text = ppd.preprocess_text(text)
    #text = ppd.remove_stopwords(text)
    print(text)
    prediction = lr_model.predict_proba([text])[0]
    print(prediction)
    print("Positive sentiment:", prediction[1])
    print("Negative sentiment:", prediction[0])

    prob_posi = prediction[1]
    prob_negi = prediction[0]

    # print the sentiment beased on the outcome of the model
    if abs(prob_posi-prob_negi) < 0.3:
        print("Neutral statment")
    elif prob_negi > 0.5:
        print("Negative sentiment")
    elif prob_posi > 0.5:
        print("Positive sentiment")

    print(prob_negi-prob_posi)
