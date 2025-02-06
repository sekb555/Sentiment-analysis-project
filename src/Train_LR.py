import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


class TrainModel:

    def __init__(self, file):
        self.file = file

    def load_data(self):
        df = pd.read_csv(self.file, encoding="utf-8")
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df

    def train_LR(self, X, Y):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42)

        pipeline = make_pipeline(
            TfidfVectorizer(), LogisticRegression(max_iter=1000)
        )

        pipeline.fit(self.X_train, self.Y_train)
        y_pred = pipeline.predict(self.X_test)

        training_accuracy = accuracy_score(
            self.Y_train, pipeline.predict(self.X_train))
        test_accuracy = accuracy_score(
            self.Y_test, pipeline.predict(self.X_test))

        print("Training accuracy:", round(training_accuracy * 100, 2), "%")
        print("Test accuracy:", round(test_accuracy * 100, 2), "%")

        return pipeline, self.X_train, self.X_test, self.Y_train, self.Y_test, y_pred

    def evaluate_model(self, con_mat):
        self.false_neg = con_mat[1][0]
        self.false_pos = con_mat[0][1]
        self.true_neg = con_mat[0][0]
        self.true_pos = con_mat[1][1]

        precision = self.true_pos / (self.true_pos + self.false_pos)
        recall = self.true_pos / (self.true_pos + self.false_neg)
        F1 = 2 * (precision * recall) / (precision + recall)

        print("Precision:", round(precision * 100, 2), "%")
        print("Recall:", round(recall * 100, 2), "%")
        print("F1 Score:", round(F1 * 100, 2), "%")

        return precision, recall, F1


TM = TrainModel("/Users/sekb/Desktop/processed_data.csv")


def main():
    df = TM.load_data()
    tweets = df['Processed_Tweets'].values
    polarity = df['Polarity'].values

    pipeline, X_train, X_test, Y_train, Y_test, y_pred = TM.train_LR(
        tweets, polarity)
    con_mat = confusion_matrix(Y_test, y_pred)
    precision, recall, F1 = TM.evaluate_model(con_mat)

    # Ensure the directory exists
    model_dir = 'data'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(pipeline, os.path.join(
        model_dir, 'logistic_regression_model.sav'))


main()
