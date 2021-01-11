from tensorflow.python.keras.metrics import Accuracy
from tensorflow.python.ops.math_ops import argmax


class CNNAccuracy(Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, 1)
        y_pred = argmax(y_pred, 1)
        return super(CNNAccuracy, self).update_state(y_true, y_pred, sample_weight)
