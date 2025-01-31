import streamlit as st
from preprocess_data import PreprocessData as ppd
import joblib
import os

lr_model = joblib.load(open('../data/logistic_regression_model.sav', 'rb'))
text = st.text_input("Enter Text you would like analyzed: ")
st.write(text)

text = ppd.preprocess_text(text)
text = ppd.remove_stopwords(text)
st.write(text)
prediction = lr_model.predict([text])
st.write(prediction)
if prediction == 0:
    st.write("Negative sentiment")
else:
    st.write("Positive sentiment")
