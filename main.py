import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import preprocessor as pp

#normalize with min/max
def normalize(data):
    return (data - np.min(data)) / (np.max(data) + 1 - np.min(data))

# Load the training data from training_data folder
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')

# for each row in the scores_csv, the first argument is the in.wav path, and the second is the out.wav, and the third is the score. Store in separate vectors:
scores_len = scores_csv.shape[0]
path = []
scores = []
for i in range(scores_len - 1):
    path = np.append(path, str(scores_csv[i][0].decode()))
    scores = np.append(scores, float(scores_csv[i][1]))

# Preprocess the input and output wav files
# data_x is a 2D array where each row is the input and output wav files concatenated in the frequency domain
# data_y is a 1D array where each element is the score of the corresponding row in data_x

data_x = [pp.preprocess_wav(librosa.load('training_data/' + path[0]))]
data_y = [[scores[0]]]

for j in range(1, scores_len - 1):
    data_x = np.append(data_x, [pp.preprocess_wav(librosa.load('training_data/' + path[j]))], axis = 0)
    data_y = np.append(data_y, [[scores[j]]], axis = 0)

# Normalize the data
data_x = normalize(data_x)
data_y = normalize(data_y)


# create sequential model
model = keras.Sequential()
model.add(layers.Input(shape=(data_x.shape[1],)))
model.add(layers.Dense(526, activation='linear'))
model.add(layers.Dense(128, activation='swish'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(16, activation='swish'))
model.add(layers.Dense(1, activation='relu'))

predictions = model(data_x)

loss_fn = keras.losses.MeanSquaredError()
loss_fn(data_y, predictions).numpy()

model.compile(optimizer='adam', loss = loss_fn, metrics=['accuracy'])

history = model.fit(data_x, data_y, epochs=10)

model.summary()

print(history.history)