import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


class TrainLR:

    def __init__(self, file, test_size=0.1, random_state=42, max_iter=1000):
        self.file = file
        self.test_size = test_size
        self.random_state = random_state
        self.max_iter = max_iter

    def load_data(self):
        df = pd.read_csv(self.file, encoding="utf-8")
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df

    def train_LR(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state
        )
        pipeline = make_pipeline(
            TfidfVectorizer(), LogisticRegression(max_iter=self.max_iter)
        )

        pipeline.fit(self.X_train, self.Y_train)
        y_pred = pipeline.predict(self.X_test)

        training_accuracy = accuracy_score(
            self.Y_train, pipeline.predict(self.X_train))
        test_accuracy = accuracy_score(
            self.Y_test, pipeline.predict(self.X_test))

        print("LR Training accuracy:", round(training_accuracy * 100, 2), "%")
        print("LR Test accuracy:", round(test_accuracy * 100, 2), "%")

        return pipeline, self.X_train, self.X_test, self.Y_train, self.Y_test, y_pred

    def evaluate_model(self, y_pred):
        print(classification_report(self.Y_test, y_pred))

    def main(self):
        df = self.load_data()
        tweets = df['Processed_Tweets'].values
        polarity = df['Polarity'].values

        pipeline, X_train, X_test, Y_train, Y_test, y_pred = self.train_LR(
            tweets, polarity)
        self.evaluate_model(y_pred)

        joblib.dump(pipeline, 'data/logistic_regression_model.sav')
