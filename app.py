import streamlit as st
from emotion import sentence_prediction


if __name__ == '__main__':
    st.set_page_config(page_title="Emotion Predictor")
    
    st.header("Emotion Predictor")

    sentence = st.text_input("Tell me something. I'll tell you what I think you're feeling")
    submit = st.button('Enter')  

    if submit:
        prediction = sentence_prediction(sentence)
        st.write('**'+prediction+'**')
        st.write("""Forgive me if that is not how you feel. I am only a humble machine learning model (not even a fancy neural network).
                """)
        st.write("""I  was created by training a machine learning model on the [Hugging Face Emotions Dataset](https://huggingface.co/datasets/emotion).""")
        st.write("""I predict the emotion expressed in the input text as Sadness, Anger, Love, Surprise, Fear or Joy.""")
        st.write("""To learn more about the nitty-gritties, check out Under the Hood.""")
        
