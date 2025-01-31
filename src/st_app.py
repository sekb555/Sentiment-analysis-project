import streamlit as st
import pickle
from preprocess_data import preprocess_text, remove_stopwords

lr_model = pickle.load(open('data/logistic_regression_model.sav', 'rb'))

text = st.text_input("Enter a tweet: ")
st.write(text)


text = ppd.preprocess_text(text)
text = ppd.remove_stopwords(text)
print(text)
