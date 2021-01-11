from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten  # Aktivacija
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.python.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D


class LSTMClassificationModel(Model):
    def __init__(self, max_nb_words, embedding_dim, input_length):
        super().__init__(self)
        self.embedding = Embedding(max_nb_words, embedding_dim, input_length=input_length)
        self.dropout = SpatialDropout1D(0.2)
        self.lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2)
        self.dense = Dense(4, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.lstm(x)
        x = self.dense(x)
        return x
