import streamlit as st

st.title("Exploratory Data Analysis")
st.write("This page shows the exploratory data analysis of the training dataset.")

st.write("### Common Phrases:")
st.image("../docs/Common_Phrases.jpg", "Frequently used phrases in the dataset")

st.write("### Word Frequency:")
st.image("../docs/word_frequency.jpg", "Top words used in the dataset")

st.write("### Sentiment Distribution:")
st.image("../docs/Sentiment_distribution.jpg", "Proportion of different sentiment labels")

st.write("### Sentiment Distribution Over Time:")
st.image("../docs/sentiment_distribution_over_time.jpg", "How sentiment trends change over time")