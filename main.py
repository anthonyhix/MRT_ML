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
from sklearn.model_selection import train_test_split


# Function to normalize data between 0 and 1 by column
def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std



# Load and preprocess data
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')
scores_len = scores_csv.shape[0]
indices = np.random.choice(scores_len, size=scores_len, replace=False)
path = scores_csv[indices, 0].astype(str)
scores = scores_csv[indices, 1].astype(float)
freq_min = 1000
freq_max = 7000
load_amt = 3500

# Preprocess wav files
# Preprocess wav files
data_x = []
data_y = []
group_size = 1
wav_file = librosa.load('training_data/' + path[0])
preprocessed_wav = pp.preprocess_wav(wav_file, sample_time = 1, group_size = group_size)
data_x.append(preprocessed_wav)
data_y.append(scores[0])
for i in range(1, min([load_amt,scores_len])):
    wav_file = librosa.load('training_data/' + path[i])
    preprocessed_wav = pp.preprocess_wav(wav_file, sample_time = 1, group_size = group_size)
    if len(preprocessed_wav) != len(data_x[0]):
        print('Error: Incorrect length of preprocessed wav file')
        continue
    data_x.append(preprocessed_wav)
    # Get the score for the wav file
    data_y.append(scores[i])
    print('Data Loaded: ' + str(i + 1) + '/' + str(min([load_amt,scores_len])))
data_x = np.stack(data_x)
data_y = np.stack(data_y)

# Plot the first row of data_x for reference
plt.plot(data_x[0])
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.figure()


# Normalize the input data
data_x = normalize(data_x)

# Compute F score for each feature

f_stat, p_val = f_classif(data_x, data_y.ravel())

# Plot F score for each feature
plt.bar(range(data_x.shape[1]), f_stat)
plt.xlabel('Feature')
plt.ylabel('F score')
plt.figure()

# Select the top n features
selector = SelectKBest(f_classif, k=350)

# Fit the selector to the training data
selector.fit(data_x, data_y.ravel())

# Transform the training and test data
data_x = selector.transform(data_x)

# Create a healthy split of data
combined = np.column_stack((data_x, data_y))

train, test = train_test_split(combined, test_size=0.2)

# Split data into training and test sets
training_data_x = train[:, :-1]
training_data_y = train[:, -1].reshape(-1, 1)
test_data_x = test[:, :-1]
test_data_y = test[:, -1].reshape(-1, 1)

# Create and compile model
model = keras.Sequential([
    layers.Input(shape=(training_data_x.shape[1],)),
    layers.Dense(24, activation = 'relu'),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(4, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])

threshold = 0.5

metrics = [
      # keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      # keras.metrics.MeanSquaredError(name='Brier score'),
      # keras.metrics.FalsePositives(name='fp'),
      keras.metrics.FalseNegatives(name='fn'), 
      # keras.metrics.BinaryAccuracy(name='accuracy'),
]
optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, 
             loss='binary_crossentropy',
              metrics=metrics)

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
plt.figure()

counts, bins = np.histogram(prediction_model.predict(test_data_x), bins=20)
plt.hist(bins[:-1], bins, weights=counts)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.figure()

print('Ratio of Likely Positives to Likely Negatives (95% Interval): ' + str(counts[-1]/counts[0]))
print('Ratio of Actual Positives to Actual Negatives (95% Interval): ' + str(np.sum(data_y)/(len(data_y) - np.sum(data_y))))


counts, bins = np.histogram(prediction_model.predict(test_data_x), bins=10)
plt.hist(bins[:-1], bins, weights=counts)
plt.xlabel('Score')
plt.ylabel('Frequency')

print('Ratio of Likely Positives to Likely Negatives (90% Interval): ' + str(counts[-1]/counts[0]))
print('Ratio of Actual Positives to Actual Negatives (90% Interval): ' + str(np.sum(data_y)/(len(data_y) - np.sum(data_y))))
plt.show()


