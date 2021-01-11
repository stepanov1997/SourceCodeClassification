from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten  # Aktivacija
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Embedding, LSTM, GlobalMaxPooling1D


class LSTMClassificationModel(Model):
    def __init__(self, train_features_size, vocabulary_size, embedded_dim, hidden_state_dim):
        super().__init__(self)
        self.embedding = Embedding(vocabulary_size+1, embedded_dim, input_shape=(train_features_size,))
        self.lstm = LSTM(hidden_state_dim, return_sequences=True)
        self.global_max_pooling = GlobalMaxPooling1D()
        self.dense = Dense(4, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.global_max_pooling(x)
        x = self.dense(x)
        return x
