#import numpy and tensorflow
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# Load the training data from training_data folder
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')

# for each row in the scores_csv, the first argument is the .wav path, and the second is the output. Store in separate vectors:
scores_len = scores_csv.shape[0]
path = []
scores = []
for i in range(scores_len):
    path = np.append(path, str(scores_csv[i][0].decode()))
    scores = np.append(scores, float(scores_csv[i][1]))
print(path)
print(scores)