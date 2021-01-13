import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from models.LSTMClassificationModel import LSTMClassificationModel

import tensorflow as tf


def calculate_accuracy(predictions, targets):
    arr = np.array([predictions[i] == targets[i] for i in range(0, len(predictions))])
    accuracy = np.count_nonzero(arr) / len(arr)
    return accuracy


classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../../data/class_indices_of_dataset.npy", allow_pickle=True).astype('uint8')
source_codes_tokenized = np.load("../../data/tokenized_data/source_codes_binary_tokenized.npy", allow_pickle=True)

X_train, X_test_val, y_train, y_test_val = train_test_split(source_codes_tokenized, class_indices, test_size=0.20)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50)


MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
model = LSTMClassificationModel(max_nb_words=MAX_NB_WORDS,
                                embedding_dim=EMBEDDING_DIM,
                                input_length=X_train.shape[1])
NUM_EPOCHS = 5
BATCH_SIZE = 1

print("[INFO] training network...")
model.compile(loss="categorical_crossentropy", optimizer=Adam(0.5), metrics=["accuracy"])

y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(y_test, len(classes))
y_val = to_categorical(y_val, len(classes))


# treniranje
model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
          verbose=1, shuffle=False, validation_data=(X_val, y_val))

model.save("../../data/fitted_classificators/lstm_binary_vect")

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
