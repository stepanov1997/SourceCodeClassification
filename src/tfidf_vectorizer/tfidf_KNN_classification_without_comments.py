import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

def calculate_accuracy(predictions, targets):
    arr = np.array([predictions[i] == targets[i] for i in range(0, len(predictions))])
    accuracy = np.count_nonzero(arr) / len(arr)
    return accuracy


classes = np.load("../../data/classes_of_dataset.npy", allow_pickle=True)
class_indices = np.load("../../data/class_indices_of_dataset.npy", allow_pickle=True).astype('uint8')
source_codes_tokenized = np.load("../../data/tokenized_data/source_codes_without_comms_tfidf_tokenized.npy", allow_pickle=True)

X_train, X_test_val, y_train, y_test_val = train_test_split(source_codes_tokenized, class_indices, test_size=0.20)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

with open("../../data/fitted_classificators/knn_tfidf_vect_without_comms.pkl", 'wb') as f:
    pickle.dump(knn, f)


predicted = knn.predict(X_test)
accuracy = calculate_accuracy(predicted, y_test)

print(accuracy)
