import streamlit as st
import os

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
