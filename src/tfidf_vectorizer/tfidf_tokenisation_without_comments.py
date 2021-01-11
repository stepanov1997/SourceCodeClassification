import pickle

import nltk as nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import util.comment_remover as comment_remover

classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../../data/class_indices_of_dataset.npy", allow_pickle=True)
source_codes = np.load("../../data/source_codes_dataset.npy", allow_pickle=True)

nltk.download('punkt')
stemmer = nltk.PorterStemmer()
minlen = 1


def tokenize(text):
    # Funkcija koja vrši tokenizaciju teksta na sastavne riječi
    text = comment_remover.remove_python_style_comments(text)

    tokens = nltk.word_tokenize(text)  # Tokenizacija teksta
    stems = []
    for token in tokens:
        stem = stemmer.stem(token)  # Stemizacija tokena
        if len(stem) > minlen:  # Zadržavanje tokena čija je dužina veća od minlen
            stems.append(stem)
    return stems


tfidef_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=tokenize)
source_codes_binary_tokenized = tfidef_vectorizer.fit_transform(source_codes)
source_codes_binary_tokenized = source_codes_binary_tokenized.toarray().astype('uint8')

with open("../../data/fitted_vectorizers/tfidf_vect_without_comms.pkl", 'wb') as f:
    pickle.dump(tfidef_vectorizer, f)
np.save("../../data/tokenized_data/source_codes_without_comms_tfidf_tokenized.npy", source_codes_binary_tokenized)


