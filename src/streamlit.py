import streamlit as st
import pandas as pd
from preprocess_data import PreprocessData
import joblib
import plotly.graph_objects as go
import os

st.sidebar.title("Navigation")
st.sidebar.write("Go to:")
page = st.sidebar.radio(
    "", ["Home", "Sentiment Analyzer", "Exploratory Data Analysis"])
if page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("This page shows the exploratory data analysis of the training dataset.")

    def get_image_path(relative_path, text):
        image_path = os.path.join(os.getcwd(), relative_path)

        if os.path.exists(image_path):
            st.image(image_path, caption=text)
        else:
            st.error(f"Image not found: {image_path}")

    st.write("### Common Phrases:")
    get_image_path("docs/Common_Phrases.jpg",
                   "Frequently used phrases in the dataset")

    st.write("### Word Frequency:")
    get_image_path("docs/word_frequency.jpg", "Top words used in the dataset")

    st.write("### Sentiment Distribution:")
    get_image_path("docs/Sentiment_distribution.jpg",
                   "Proportion of different sentiment labels")

    st.write("### Sentiment Distribution Over Time:")
    get_image_path("docs/sentiment_distribution_over_time.jpg",
                   "How sentiment trends change over time")

elif page == "Sentiment Analyzer":

    ppd = PreprocessData()

    st.title("Sentiment Analysis App")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    option = st.selectbox(
        "How would you like to be contacted?",
        ("Logistic Regression", "Naive Bayes")
    )

    if option == "Logistic Regression":
        model_path = os.path.join(
            base_dir, "../data/logistic_regression_model.sav")
    elif option == "Naive Bayes":
        model_path = os.path.join(base_dir, "../data/NB_model.sav")

    model = joblib.load(model_path)  # Load model
    if option == "Logistic Regression":
        st.write("Logistic Regression Model")
    elif option == "Naive Bayes":
        st.write("Naive Bayes Model")

    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = {
            'Positive': 0, 'Negative': 0, 'Neutral': 0}

    text = st.text_input("### Enter a text to analyze:")

    prob_negi = 0
    prob_posi = 0
    if text.strip() == "":
        st.write("Nothing entered. Please enter some text.")
    else:
        text = ppd.preprocess_text(text)
        # text = ppd.remove_stopwords(text)
        prediction = model.predict_proba([text])[0]
        prob_negi = prediction[0]
        prob_posi = prediction[1]

        if abs(prob_posi-prob_negi) < 0.2:
            sentiment = "Neutral statement"
            st.session_state.sentiment['Neutral'] += 1
            st.write(sentiment)
        elif prob_negi > 0.5:
            sentiment = "Negative sentiment"
            st.session_state.sentiment['Negative'] += 1
            st.write(sentiment)
        elif prob_posi > 0.5:
            sentiment = "Positive sentiment"
            st.session_state.sentiment['Positive'] += 1
            st.write(sentiment)
            
    st.write("Positive statements:", st.session_state.sentiment['Positive'], "Neutral statements:",
             st.session_state.sentiment['Neutral'], "Negative statements:", st.session_state.sentiment['Negative'])

    if st.button("Reset"):
        st.session_state.sentiment = {
            'Positive': 0, 'Negative': 0, 'Neutral': 0}

elif page == "Home":
    st.title("Home")
    st.write("Welcome to my Sentiment Analysis App!")
    st.write(
        "This app uses a logistic regression model to predict the sentiment of a given text.")
