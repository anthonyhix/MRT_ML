import numpy as np
from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import librosa
import preprocessor as pp

# Load the training data from training_data folder
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')

# for each row in the scores_csv, the first argument is the in.wav path, and the second is the out.wav, and the third is the score. Store in separate vectors:
scores_len = scores_csv.shape[0]
path_in = []
path_out = []
scores = []
for i in range(scores_len):
    path_in = np.append(path_in, str(scores_csv[i][0].decode()))
    path_out = np.append(path_out, str(scores_csv[i][1].decode()))
    scores = np.append(scores, float(scores_csv[i][2]))
print(path_in)
print(path_out)
print(scores)

# Preprocess the input and output wav files
# data_x is a 2D array where each row is the input and output wav files concatenated in the frequency domain
# data_y is a 1D array where each element is the score of the corresponding row in data_x

data_x = [np.append(pp.preprocess_wav(librosa.load('training_data/' + path_in[0])), pp.preprocess_wav(librosa.load('training_data/' + path_out[0])))]
data_y = [[scores[0]]]

for j in range(1, scores_len):
    data_x = np.append(data_x, [np.append(pp.preprocess_wav(librosa.load('training_data/' + path_in[j])), pp.preprocess_wav(librosa.load('training_data/' + path_out[j])))], axis = 0)
    data_y = np.append(data_y, [[scores[j]]], axis = 0)
    