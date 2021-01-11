import pickle

import nltk
import numpy as np

classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True).tolist()


def calculate_accuracy(predictions, targets):
    arr = np.array([predictions[i] == targets[i] for i in range(0, len(predictions))])
    accuracy = np.count_nonzero(arr) / len(arr)
    return accuracy


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


with open("../../data/fitted_classificators/svm_tfidf_vect.pkl", 'rb') as f:
    svm = pickle.load(f)
with open("../../data/fitted_vectorizers/tfidf_vect.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

#f = open("./src/dataset_maker.py", "r", encoding='UTF-8')
example = vectorizer.transform(["System.out.println(""Hello world"")"])
predicted = svm.predict(example.toarray())
print(f'Prediction for Java code: {classes[predicted[0]]}')
