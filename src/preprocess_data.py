import html
import pandas as pd
import numpy as np


def preprocess_text(text):
    text = text.astype(str).str.replace(
        r'@\w+', '', regex=True)  # remove user tags
    text = text.astype(str).str.replace(
        r'http\S+', '', regex=True)  # remove URLs
    text = text.astype(str).str.replace(
        r'[^a-zA-Z0-9 ]', '', regex=True)  # remove special characters
    # remove leading and trailing whitespaces
    text = text.astype(str).str.strip()
    text = text.apply(html.unescape)  # remove html
    return text


def process_date(date):
    date = date.astype(str).str.split()
    date = pd.DataFrame(date.tolist(), columns=[
                        'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
    date = date.drop(columns=['Timezone', 'Time'])
    return date


# read the training data and assign the columns names
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv", header=None, encoding="ISO-8859-1")
df.columns = ["Polarity", "ID", "Date", "Flag", "User", "Tweet"]

df.drop(columns=["ID", "Flag", "User"], inplace=True)

# assign text and sentiment to variables
twts = df["Tweet"]
sentiments = df["Polarity"]
sentiments = (sentiments == 4).astype(int)  # 0 for negative, 1 for positive
df["Polarity"] = sentiments


# preprocess the input data 
df['Processed_Tweets'] = preprocess_text(twts)
df.drop(columns=["Tweet"], inplace=True) 

# split the date column and remove unnecessary columns
dates = df["Date"]
dates_split = dates.astype(str).str.split()
dates = pd.DataFrame(dates_split.tolist(), columns=[
                     'Day', 'Month', 'Date', 'Time', 'Timezone', 'Year'])
dates = dates.drop(columns=['Timezone', 'Time', 'Day'])


df["DateOnly"] = pd.to_datetime(
    dates['Month'] + " " + dates['Date'] + " " + dates['Year'])
grouped = df.groupby("DateOnly")[
    'Polarity'].value_counts().unstack().fillna(0)

df.to_csv("data/processed_data.csv", index=False)
