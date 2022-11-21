import pickle
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

spacy_stopwords = pickle.load(open('model_spacy_stopwords.pkl','rb'))
nlp = pickle.load(open('model_spacy_nlp.pkl','rb'))
vectorizer = pickle.load(open('model_vectorizer.pkl','rb'))
model = pickle.load(open('model_lgbm_classifier.pkl','rb'))

classes = model.classes_

def lemmatize(s):
    return ' '.join([str(word) for word in nlp(s) if str(word.lemma_).lower() not in spacy_stopwords])


def sentence_prediction(sentence):
    y_predict_proba = model.predict_proba(vectorizer.transform([lemmatize(sentence)]).todense())[0]
    args = np.flip(np.argsort(y_predict_proba))
    emotion_pred = classes[args]
    prob_pred = np.flip(np.sort(y_predict_proba))
    if(prob_pred[0]>0.9):
        prediction = "There is a {:.2f}% chance you feel {}".format(prob_pred[0]*100,emotion_pred[0])
    else:
        prediction="There is a {:.2f}% chance you feel {} and a {:.2f}% chance you feel {}".format(prob_pred[0]*100,emotion_pred[0],
                                                                                         prob_pred[1]*100,emotion_pred[1])
    return prediction
    
