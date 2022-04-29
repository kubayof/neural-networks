import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# number of words in data
num_words = 5000


def load_data():
    all_data = np.load('lab3_data/data.npy', allow_pickle=True)
    x_train_np = all_data[0]
    y_train_np = all_data[1]
    x_test_np = all_data[2]
    y_test_np = all_data[3]
    return x_train_np, y_train_np, x_test_np, y_test_np


x_train, y_train, x_test, y_test = load_data()

max_review_length = 500
X_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

embedding_vecor_length = 32
model = Sequential(
    Embedding(num_words, embedding_vecor_length, input_length=max_review_length),
    LSTM(100),
    Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

