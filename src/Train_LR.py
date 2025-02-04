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


def load_data():
    df = pd.read_csv("/Users/sekb/Desktop/processed_data.csv",
                     encoding="ISO-8859-1")
    df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
    return df


def train_LR(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    pipeline = make_pipeline(
        TfidfVectorizer(), LogisticRegression(max_iter=1000))

    pipeline.fit(X_train, Y_train)

    y_pred = pipeline.predict(X_test)

    return pipeline, X_train, X_test, Y_train, Y_test, y_pred


def evaluate_model(con_mat):
    false_neg = con_mat[1][0]
    false_pos = con_mat[0][1]
    true_neg = con_mat[0][0]
    true_pos = con_mat[1][1]

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    F1 = 2*(precision*recall)/(precision+recall)

    return precision, recall, F1


def main():
    df = load_data()
    tweets = df['Processed_Tweets'].values
    polarity = df['Polarity'].values

    pipeline, X_train, X_test, Y_train, Y_test, y_pred = train_LR(
        tweets, polarity)
    con_mat = confusion_matrix(Y_test, y_pred)
    precision, recall, F1 = evaluate_model(con_mat)

    training_accuracy = accuracy_score(Y_train, pipeline.predict(X_train))
    test_accuracy = accuracy_score(Y_test, pipeline.predict(X_test))

    print("Precision:", round(precision*100, 2), "%")
    print("Recall:", round(recall*100, 2), "%")
    print("F1 Score:", round(F1*100, 2), "%")

    print('Training accuracy: ', round(training_accuracy*100, 2), "%")
    print('Test accuracy: ', round(test_accuracy*100, 2), "%")

    joblib.dump(pipeline, open('data/logistic_regression_model.sav', 'wb'))


main()
