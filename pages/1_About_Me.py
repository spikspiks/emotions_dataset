import streamlit as st

st.set_page_config(page_title="About Me")
st.markdown("""
This emotion predictor has been created by training a machine learning model on the [Hugging Face Emotions Dataset](https://huggingface.co/datasets/emotion).\n
It predicts the emotion expressed in the input text as Sadness, Anger, Love, Surprise, Fear or Joy.\n
To learn more about the nitty-gritties, check out Under the Hood.
""")