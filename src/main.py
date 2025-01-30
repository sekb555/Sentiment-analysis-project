import pandas as pd
import numpy as np
from collections import Counter
import re

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

BATCH_SIZE = 64
MAX_EPOCHS = 10
CHUNK_SIZE = 170000
VOCAB_SIZE = 922770
MAX_LEN = 32


df = pd.read_csv("data/processed_data.csv", encoding="ISO-8859-1")
df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)


