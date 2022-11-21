import streamlit as st
from emotion import sentence_prediction

st.set_page_config(page_title="Emotion Predictor")

#sentence = st.text_input("Tell me how you feel")

#prediction = sentence_prediction
sentence = st.text_input("Tell me something, I'll tell you what I think you're feeling")
submit = st.button('Enter')  

if submit:
    prediction = sentence_prediction(sentence)
    st.write(prediction)
    st.write("""Forgive me if that is not how you feel. I am only a humble machine learning model (not even a fancy neural network).
             If you want to know more about how I work, check out Under the Hood.""")