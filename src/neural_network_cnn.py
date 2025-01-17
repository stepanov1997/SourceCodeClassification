# -*- coding: utf-8 -*-
"""NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19ZwN2N2zbOfuZOhKLYqrasRcos_vONQd
"""
from models.CNNClassificationModel import CNNClassificationModel
import pickle
import json

ROOT_PATH = '..'

# @
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

classes = np.load(f"{ROOT_PATH}/data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load(f"{ROOT_PATH}/data/class_indices_of_dataset.npy", allow_pickle=True).astype('uint8')
source_codes_tokenized = np.load(f"{ROOT_PATH}/data/tokenized_data/source_codes_tokenized.npy", allow_pickle=True)

i = 1
vocabulary = {}
for words in source_codes_tokenized:
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = i
            i += 1

json_voc = json.dumps(vocabulary)
with open("../data/vocabulary2.json", "w") as f:
    f.write(json_voc)


from sklearn.model_selection import train_test_split

X_train_tokenized, X_test_val, y_train, y_test_val = train_test_split(source_codes_tokenized, class_indices,
                                                                      test_size=0.20)
X_test_tokenized, X_val_tokenized, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50)


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


train_encoded = [encode_text(doc, vocabulary) for doc in X_train_tokenized]
test_encoded = [encode_text(doc, vocabulary) for doc in X_test_tokenized]
val_encoded = [encode_text(doc, vocabulary) for doc in X_val_tokenized]

onehot_encoder = OneHotEncoder(sparse=False)

y_train = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(y_train)

y_test = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.transform(y_test)

y_val = y_val.reshape(len(y_val), 1)
y_val = onehot_encoder.transform(y_val)

lens = []
for doc in train_encoded:
    lens.append(len(doc))

from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dopunjavanje do potrebne dužine
train_padded = pad_sequences(train_encoded, maxlen=np.max(lens), padding='post')
test_padded = pad_sequences(test_encoded, maxlen=np.max(lens), padding='post')
val_padded = pad_sequences(val_encoded, maxlen=np.max(lens), padding='post')

### KOD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = CNNClassificationModel(vocab_size=len(vocabulary))

### KOD
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# definisanje optimizatora
opt = Adam(lr=1e-3)
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# treniranje
with tf.device("/device:GPU:0"):
    history = model.fit(train_padded, y_train,
                        epochs=10, batch_size=128, verbose=1, shuffle=True,
                        validation_data=(val_padded, y_val))

model.save(f"{ROOT_PATH}/data/fitted_neural_networks/code_classification_neural_network_cnn")
with open(f"{ROOT_PATH}/data/fitted_neural_networks/code_classification_neural_network_cnn_history.json", 'w') as file:
    json.dump(history.history, file)

import matplotlib.pyplot as plt
import numpy as np

N = np.arange(0, 10)
title = "Training Loss and Accuracy on source code classification (CNN)"

plt.style.use("ggplot")
plt.figure()
plt.plot(N, history.history["loss"], label="train_loss")
plt.plot(N, history.history["val_loss"], label="val_loss")
plt.plot(N, history.history["accuracy"], label="train_acc")
plt.plot(N, history.history["val_accuracy"], label="val_acc")
plt.title(title)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

results = model.evaluate(test_padded, y_test, batch_size=128)
with open(f"{ROOT_PATH}/data/fitted_neural_networks/code_classification_neural_network_cnn_test_history.json", 'w') as file:
    json.dump(results, file)