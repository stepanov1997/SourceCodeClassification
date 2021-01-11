import pickle

import nltk as nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../../data/class_indices_of_dataset.npy", allow_pickle=True)
source_codes = np.load("../../data/source_codes_dataset.npy", allow_pickle=True)

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


tfidef_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenize)
source_codes_tokenized = tfidef_vectorizer.fit_transform(source_codes)
source_codes_tokenized = source_codes_tokenized.toarray().astype('float')

with open("../../data/fitted_vectorizers/tfidf_vect.pkl", 'wb') as f:
    pickle.dump(tfidef_vectorizer, f)
np.save("../../data/tokenized_data/source_codes_tfidf_tokenized.npy", source_codes_tokenized)


