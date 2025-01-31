import pandas as pd
import joblib


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def logistic_regression(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    pipeline = make_pipeline(
        TfidfVectorizer(), LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, Y_train)

    training_accuracy = accuracy_score(Y_train, pipeline.predict(X_train))
    test_accuracy = accuracy_score(Y_test, pipeline.predict(X_test))

    return confusion_matrix(Y_test, pipeline.predict(X_test)), training_accuracy, test_accuracy, pipeline


df = pd.read_csv("data/processed_data.csv", encoding="ISO-8859-1")
df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)

tweets = df['Processed_Tweets'].values
polarity = df['Polarity'].values

con_mat, training_accuracy, test_accuracy, model = logistic_regression(
    tweets, polarity)
false_neg = con_mat[1][0]
false_pos = con_mat[0][1]
true_neg = con_mat[0][0]
true_pos = con_mat[1][1]

"""Evaluation Metrics"""
precision = true_pos/(true_pos+false_pos)
recall = true_pos/(true_pos+false_neg)
F1 = 2*(precision*recall)/(precision+recall)
print("Precision:", round(precision*100, 2), "%")
print("Recall:", round(recall*100, 2), "%")
print("F1 Score:", round(F1*100, 2), "%")

print('Training accuracy: ', round(training_accuracy*100, 2), "%")
print('Test accuracy: ', round(test_accuracy*100, 2), "%")


"""Download the model"""
filename = 'data/logistic_regression_model.sav'
joblib.dump(model, open(filename, 'wb'))
