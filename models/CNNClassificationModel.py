from numpy import argmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten  # Aktivacija
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Dropout
from tensorflow.python.keras.metrics import Accuracy


class CNNClassificationModel(Model):
    def __init__(self, vocab_size, embedding_dim=20):
        super().__init__(self)
        self.embedding = Embedding(vocab_size + 1, embedding_dim)
        self.conv1d_32 = Conv1D(32, 3, activation='relu')
        self.conv1d_64 = Conv1D(64, 3, activation='relu')
        self.conv1d_128 = Conv1D(128, 3, activation='relu')
        self.max_pooling = MaxPooling1D(4)
        self.global_max_pooling = GlobalMaxPooling1D()
        self.dense = Dense(4, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        x = self.conv1d_32(x)
        x = self.max_pooling(x)
        x = self.conv1d_64(x)
        x = self.max_pooling(x)
        x = self.conv1d_128(x)
        x = self.global_max_pooling(x)
        x = self.dense(x)
        return x
