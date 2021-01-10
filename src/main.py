import pickle

import numpy as np

classes = np.load("../data/classes_of_dataset.npy", allow_pickle=True).tolist()


def calculate_accuracy(predictions, targets):
    arr = np.array([predictions[i] == targets[i] for i in range(0, len(predictions))])
    accuracy = np.count_nonzero(arr) / len(arr)
    return accuracy


with open("../data/fitted_classificators/knn_count_vect.pkl", 'rb') as f:
    knn = pickle.load(f)
with open("../data/fitted_classificators/count_vect.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

php = vectorizer.transform("<?php></php>")
predicted = knn.predict(php)
print(classes[predicted[0]])
