import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


class TrainNB:

    def __init__(self, file, test_size=0.1, random_state=42):
        self.file = file
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        try:
            df = pd.read_csv(self.file, encoding="utf-8")
        except:
            print("Error loading data from", self.file)
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df

    def train_NB(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.random_state)

        pipeline = make_pipeline(
            TfidfVectorizer(), MultinomialNB()
        )

        pipeline.fit(self.X_train, self.Y_train)
        y_pred = pipeline.predict(self.X_test)

        training_accuracy = accuracy_score(
            self.Y_train, pipeline.predict(self.X_train))
        test_accuracy = accuracy_score(
            self.Y_test, pipeline.predict(self.X_test))

        print("NB Training accuracy:", round(training_accuracy * 100, 2), "%")
        print("NB Test accuracy:", round(test_accuracy * 100, 2), "%")

        return pipeline, self.X_train, self.X_test, self.Y_train, self.Y_test, y_pred

    def evaluate_model(self, y_pred):
        print(classification_report(self.Y_test, y_pred))

    def main(self):
        df = self.load_data()

        tweets = df['Processed_Tweets'].values
        polarity = df['Polarity'].values
        pipeline, X_train, X_test, Y_train, Y_test, y_pred = self.train_NB(
            tweets, polarity)
        self.evaluate_model(y_pred)

        joblib.dump(pipeline, "data/NB_model.sav")
