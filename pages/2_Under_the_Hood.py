import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


st.markdown("""This is a multiclass classification problem (the 6 classes being Sadness, Anger, Love, Surprise, Fear, Joy). 
            On top of that, the classes are imbalanced.""")
image = Image.open('pie.png')
st.image(image)

model_performance = pd.DataFrame(columns=['Classifier','Trainig Time','Model Size'])
model_performance.loc[0] = ["Decision Tree Classifier","8 min 38 sec","391 KB"]
model_performance.loc[1] = ["Random Forest Classifier","4 min 25 sec","91.4 MB"]
model_performance.loc[2] = ["LightGBM Classifier","4 sec","2.32 MB"]


st.markdown("""
A very simple approach was used - after lemmatizing the text, the text was vectorized using TF-IDF vectorizer and 
then [LightGBM Classifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) was fit to the dataset. \n
Why LightGBM? After trying several models, (like Decision Tree and Random Forest) LightGBM emerged as the one which had lowest training time 
and best performance.\n
The objective was to build a simple model to achieve good recall scores across all classes. \n
To be fair, Light GBM, Decision Tree and Random Forest, all achieved comparable scores on all metrics (ROC-AUC score, recall scores). 
Light GBM took the smallest time to train (by a large margin), and the model size is small.""")
st.table(model_performance)  
st.markdown("Check out the source code and the jupyter notebook describing the model-building, and the limitations of the model, at my [github repository](https://github.com/spikspiks/emotions_dataset).")

