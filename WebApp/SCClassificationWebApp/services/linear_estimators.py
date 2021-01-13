import os
import pickle
import re

import nltk
import numpy as np
ROOT = "C:\\Users\\stepa\\PycharmProjects\\SCClassification"
print()



def classificate(text, type_of_estimator='knn', type_of_vectorization='binary', representation_without_comments=False):
    text = str(text.read(),encoding='UTF-8')
    text = remove_python_style_comments(text) if representation_without_comments else text
    if type_of_vectorization.lower() not in ['binary', 'count', 'tfidf'] or type_of_estimator.lower() not in ['knn','svm']:
        raise NotImplementedError
    type_of_vectorization, type_of_estimator = type_of_vectorization.lower(), type_of_estimator.lower()

    classes = np.load(f"{ROOT}\\data\\classes_of_dataset.npy", allow_pickle=True).tolist()
    svm_cf_filename = f"{ROOT}\\data\\fitted_classificators\\{type_of_estimator}_{type_of_vectorization}_vect{'_without_comms' if representation_without_comments else ''}.pkl"
    with open(svm_cf_filename, 'rb') as f:
        estimator = pickle.load(f)
    vectorizer_filename = f"{ROOT}\\data\\fitted_vectorizers\\{type_of_vectorization}_vect{'_without_comms' if representation_without_comments else ''}.pkl"
    with open(vectorizer_filename, 'rb') as f:
        vectorizer = pickle.load(f)

    vectorized_text = vectorizer.transform([text])
    predicted = estimator.predict(vectorized_text.toarray() if type_of_estimator == 'svm' else vectorized_text)
    return classes[predicted[0]]


def remove_python_style_comments(text):
    new_text = text
    regex = re.compile(r"((\/\*)|(\"\"\"))[\s\S]*?((\*\/)|(\"\"\"))|([^:]|^)((\/\/)|#).*?$", re.MULTILINE | re.DOTALL)

    while True:
        old = new_text
        new_text = regex.sub("", new_text)

        if old == new_text:
            break
    return new_text


nltk.download('punkt')
stemmer = nltk.PorterStemmer()
minlen = 1

def tokenize(text):
    # Funkcija koja vrši tokenizaciju teksta na sastavne riječi
    tokens = nltk.word_tokenize(text)  # Tokenizacija teksta
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)  # Stemizacija tokena
        if len(stem) > minlen:  # Zadržavanje tokena čija je dužina veća od minlen
            stems.append(stem)
    return stems