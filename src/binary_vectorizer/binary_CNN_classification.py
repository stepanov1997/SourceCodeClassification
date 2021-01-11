
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from models.CNNAccuracy import CNNAccuracy

from models.CNNClassificationModel import CNNClassificationModel
from models.LSTMClassificationModel import LSTMClassificationModel

from keras.optimizers import SGD

def calculate_accuracy(predictions, targets):
    arr = np.array([predictions[i] == targets[i] for i in range(0, len(predictions))])
    accuracy = np.count_nonzero(arr) / len(arr)
    return accuracy


classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../../data/class_indices_of_dataset.npy", allow_pickle=True).astype('uint8')
source_codes_tokenized = np.load("../../data/tokenized_data/source_codes_binary_tokenized.npy", allow_pickle=True)

X_train, X_test_val, y_train, y_test_val = train_test_split(source_codes_tokenized, class_indices, test_size=0.20)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50)

model = CNNClassificationModel(X_train.shape[1])

# veličina batch-a
BATCH_SIZE = 20
# broj epoha koji se trenira
NUM_EPOCHS = 30

# definisanje optimizatora
opt = SGD(lr=0.01)
print("[INFO] training network...")

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

# treniranje
y_train = to_categorical(y_train, len(classes))
y_test = to_categorical(y_test, len(classes))
y_val = to_categorical(y_val, len(classes))


model.fit(X_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(X_val, y_val))

model.save("../../data/fitted_classificators/lstm_binary_vect")

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

