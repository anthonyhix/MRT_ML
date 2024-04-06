import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import pandas as pd
import preprocessor as pp

#normalize with min/max
def normalize(data):
    for i in range(data.shape[0]):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))
    return data

# Load the training data from training_data folder
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')

# for each row in the scores_csv, the first argument is the in.wav path, and the second is the out.wav, and the third is the score. Store in separate vectors:
scores_len = scores_csv.shape[0]
path = []
scores = []
batch_size = 1000
for i in range(np.min([scores_len, batch_size])):
    path = np.append(path, str(scores_csv[i][0].decode()))
    scores = np.append(scores, float(scores_csv[i][1]))

# Preprocess the input and output wav files
# data_x is a 2D array where each row is the input and output wav files concatenated in the frequency domain
# data_y is a 1D array where each element is the score of the corresponding row in data_x

indices = np.arange(batch_size)
np.random.shuffle(indices)

path = path[indices]
scores = scores[indices]

data_x = [pp.preprocess_wav(librosa.load('training_data/' + path[0]))]
data_y = [[scores[0]]]
labels = [0, 1]

for j in range(1, np.min([scores_len, batch_size])):
    data_x = np.append(data_x, [pp.preprocess_wav(librosa.load('training_data/' + path[j]), end_offset = 1)], axis = 0)
    data_y = np.append(data_y, [[scores[j]]], axis = 0)
    print('\nPreprocessed ' + str(j) + ' of ' + str(np.min([scores_len, batch_size])))

# Normalize the input data
data_x = normalize(data_x)

# create sequential model
model = keras.Sequential()
model.add(layers.Input(shape=(data_x.shape[1],)))
model.add(layers.Dense(526, activation = 'relu'))
model.add(layers.Dense(128))
model.add(layers.Dense(64))
model.add(layers.Dense(16))
model.add(layers.Dense(2))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer='adam', loss = loss_fn, metrics=['accuracy'])

history = model.fit(data_x, data_y, epochs=40)

test_loss, test_acc = model.evaluate(data_x, data_y, verbose=1)

print('\nTest accuracy:', test_acc)
print('\nPercent of 0:', np.sum(data_y == 0) / len(data_y) * 100)
print('\nPercent of 1:', np.sum(data_y == 1) / len(data_y) * 100)

model.summary()

# predict
model.add(layers.Softmax())

output = np.argmax(model.predict(data_x))

df = pd.DataFrame({'Actual': data_y, 'Predicted': output})

df.to_csv('output.csv')

print(history.history)