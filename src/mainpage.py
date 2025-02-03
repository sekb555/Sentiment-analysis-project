import streamlit as st
import pandas as pd
from preprocess_data import PreprocessData
import joblib
import plotly.graph_objects as go
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "../data/logistic_regression_model.sav")

lr_model = joblib.load(model_path)  # Load model

ppd = PreprocessData()

st.title("Sentiment Analysis App")
st.write("This app uses a logistic regression model to predict the sentiment of a given text.")

if 'sentiment' not in st.session_state:
    st.session_state.sentiment = {'Positive': 0, 'Negative': 0, 'Neutral': 0}


text = st.text_input("### Enter a text to analyze:")

prob_negi = 0
prob_posi = 0
if st.button("Analyze"):
    if text.strip() == "":
        st.write("Nothing entered. Please enter some text.")
    else:
        text = ppd.preprocess_text(text)
        text = ppd.remove_stopwords(text)
        prediction = lr_model.predict_proba([text])[0]
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

col1, col2 = st.columns([20, 3], vertical_alignment="bottom")


with col1:
    st.write("Positive statements:", st.session_state.sentiment['Positive'], "Neutral statements:",
         st.session_state.sentiment['Neutral'], "Negative statements:", st.session_state.sentiment['Negative'])
    st.write("## Sentiment Analysis Results")
    if st.session_state.sentiment != {'Positive': 0, 'Negative': 0, 'Neutral': 0}:

        labels = ["Positive", "Negative", "Neutral"]
        values = [st.session_state.sentiment[label] for label in labels]
        colors = ["#2ECC71", "#E74C3C", "#3498DB"]

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("### No data to display")

with col2:
    if st.button("Reset"):
        st.session_state.sentiment = {
            'Positive': 0, 'Negative': 0, 'Neutral': 0}
