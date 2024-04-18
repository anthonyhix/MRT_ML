import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import preprocessor as pp
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif, chi2


# Function to normalize data between 0 and 1 by column
def normalize(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data = (data - min_vals) / (max_vals - min_vals)
    return data


# Load and preprocess data
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')
scores_len = scores_csv.shape[0]
indices = np.random.choice(scores_len, size=scores_len, replace=False)
path = scores_csv[indices, 0].astype(str)
scores = scores_csv[indices, 1].astype(float)
freq_min = 1000
freq_max = 7000

# Preprocess wav files
# Preprocess wav files
data_x = []
data_y = []
group_size = 1
wav_file = librosa.load('training_data/' + path[0])
preprocessed_wav = pp.preprocess_wav(wav_file, sample_time = 0.5, group_size = group_size)
data_x.append(preprocessed_wav)
data_y.append(scores[0])
for i in range(1, scores_len):
    wav_file = librosa.load('training_data/' + path[i])
    preprocessed_wav = pp.preprocess_wav(wav_file, sample_time = 0.5, group_size = group_size)
    if len(preprocessed_wav) != len(data_x[0]):
        print('Error: Incorrect length of preprocessed wav file')
        continue
    data_x.append(preprocessed_wav)
    # Get the score for the wav file
    data_y.append(scores[i])
data_x = np.stack(data_x)
data_y = np.stack(data_y)

# Plot the first row of data_x for reference
plt.plot(data_x[0])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.figure()


# Normalize the input data
#data_x = normalize(data_x)

# Compute F score for each feature

f_stat, p_val = f_classif(data_x, data_y.ravel())

# Plot F score for each feature
plt.bar(range(data_x.shape[1]), f_stat)
plt.xlabel('Feature')
plt.ylabel('F score')
plt.figure()

# Select the top n features
selector = SelectKBest(f_classif, k=75)

# Fit the selector to the training data
selector.fit(data_x, data_y.ravel())

# Transform the training and test data
data_x = selector.transform(data_x)


# Split data into training and test sets
training_batch_size = 3000
test_batch_size = 500
training_data_x = data_x[:training_batch_size, :]
training_data_y = data_y[:training_batch_size]
test_data_x = data_x[training_batch_size:, :]
test_data_y = data_y[training_batch_size:]

# Create and compile model
model = keras.Sequential([
    layers.Input(shape=(data_x.shape[1],)),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])

threshold = 0.5

binary_metric = tf.keras.metrics.BinaryIoU(threshold=threshold)
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, 
             loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# Train model
history = model.fit(training_data_x, training_data_y, epochs=100, validation_split = 0.1)

# Create prediction model
prediction_model = model
# if output is greater than threshold, set to 1, else set to 0

output = prediction_model.predict(test_data_x) > threshold
model.summary()

# Create dataframe for results
df = pd.DataFrame({
    'actual': test_data_y.ravel(),
    'predicted': output.ravel()
})

# Add columns for false negatives, true negatives, true positives, and false positives
for condition, label in zip([(1, 0), (0, 0), (1, 1), (0, 1)], 
                            ['False Negative', 'True Negative', 'True Positive', 'False Positive']):
    df[label] = (df['actual'] == condition[0]) & (df['predicted'] == condition[1])

# Plot results
plot = df.iloc[:, 2:].sum().plot(kind='bar')
plt.xticks(rotation=0)
plt.figure()

# Plot bar graphs for amount of 1's and 0's in the actual and predicted columns on the same plot with bars next to each other
df['actual'].value_counts().plot(kind='bar', color='forestgreen', position=0, width=0.25)
df['predicted'].value_counts().plot(kind='bar', color='indianred', position=1, width=0.25)
# change x axis labels to true/false
plt.xticks([0, 1], ['True', 'False'], rotation=0)
plt.xticks(rotation=0)
plt.legend(['Actual', 'Predicted'])
plt.figure()

# Plot validation loss vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower left')
plt.show()
