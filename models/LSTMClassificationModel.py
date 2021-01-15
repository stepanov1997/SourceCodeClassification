from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten  # Aktivacija
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D, \
    GlobalAveragePooling1D, Dropout


class LSTMClassificationModel(Model):
    def __init__(self, max_words, max_len, classes_size):
        super().__init__(self)
        self.embedding = Embedding(max_words+1, 50, input_length=max_len)
        self.lstm = LSTM(64)
        self.dense = Dense(256, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(classes_size, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
