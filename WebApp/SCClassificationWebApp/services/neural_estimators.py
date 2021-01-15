import json
import os
import pickle
import re
import warnings

import nltk
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

ROOT = "C:\\Users\\stepa\\PycharmProjects\\SCClassification"


def classificate(text, type_of_neural_network='cnn'):
    text = str(text.read(), encoding='UTF-8')
    type_of_neural_network = type_of_neural_network.lower()
    if type_of_neural_network not in ['dnn', 'cnn']:
        raise NotImplementedError

    classes = np.load(f"{ROOT}\\data\\classes_of_dataset.npy", allow_pickle=True).tolist()
    fitted_model = load_model(
        f"{ROOT}\\data\\fitted_neural_networks\\code_classification_neural_network_{'cnn' if type_of_neural_network == 'cnn' else 'dense'}")

    with open(f"{ROOT}\\data\\vocabulary2.json", 'rb') as f:
        vocabulary = json.load(f)

    text = tokenize(text)
    text = encode_text(text, vocabulary)

    max_length = fitted_model.trainable_variables[0].shape[0]
    text_padded = pad_sequences([text], maxlen=247353, padding='post')

    res = (fitted_model.predict(text_padded) > 0.5).astype("int32")
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoder.fit(np.array([0, 1, 2, 3]).reshape(4, 1))
    predicted = onehot_encoder.inverse_transform(list(res))

    return classes[predicted[0, 0]]


nltk.download('punkt')
stemmer = nltk.PorterStemmer()
minlen = 1


def encode_text(doc, vocabulary):
    '''Funkcija koja vrši enkodovanje dokumenta u listu cjelobrojnih vrijednosti.
  Ukoliko za riječ ne postoji maprianje onda je preskočiti.
  Ulazni argumenti su:
  doc - list tokena iz dokumenta
  vocabulary - mapriranje riječ->indeks
  Izlaz treba da bude list cjelobrojnih vrijednosti.'''
    res = []
    for word in doc:
        if word in vocabulary:
            res.append(vocabulary[word])
    return res


def tokenize(text):
    # Funkcija koja vrši tokenizaciju teksta na sastavne riječi
    tokens = nltk.word_tokenize(text)  # Tokenizacija teksta
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)  # Stemizacija tokena
        if len(stem) > minlen:  # Zadržavanje tokena čija je dužina veća od minlen
            stems.append(stem)
    return stems
