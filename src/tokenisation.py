import pickle

import nltk as nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

classes = np.load("../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../data/class_indices_of_dataset.npy", allow_pickle=True)
source_codes = np.load("../data/source_codes_dataset.npy", allow_pickle=True)

nltk.download('punkt')
minlen = 1


def tokenize(text):
    # Funkcija koja vrši tokenizaciju teksta na sastavne riječi
    tokens = nltk.word_tokenize(text)  # Tokenizacija teksta
    stems = []
    for token in tokens:
        if len(token) > minlen:  # Zadržavanje tokena čija je dužina veća od minlen
            stems.append(token)
    return stems


tokenized = np.array([tokenize(source_code) for source_code in source_codes])

np.save("../data/tokenized_data/source_codes_tokenized.npy", tokenized)
