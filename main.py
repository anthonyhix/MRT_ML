import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import pandas as pd
import matplotlib.pyplot as plt
import preprocessor as pp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Function to normalize data between 0 and 1 by column
def normalize(data):
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))
    return data

# Load and preprocess data
scores_csv = np.loadtxt('training_data/scores.csv', delimiter=',', dtype = 'S')
scores_len = scores_csv.shape[0]
indices = np.random.choice(scores_len, size=scores_len, replace=False)
path = scores_csv[indices, 0].astype(str)
scores = scores_csv[indices, 1].astype(float)

# Preprocess wav files
data_x = np.array([pp.preprocess_wav(librosa.load('training_data/' + p)) for p in path])
data_y = scores[:, np.newaxis]

# Normalize the input data
#data_x = normalize(data_x)

# Split data into training and test sets
training_batch_size = 3000
test_batch_size = 500
training_data_x = data_x[:training_batch_size, :]
training_data_y = data_y[:training_batch_size, :]
test_data_x = data_x[training_batch_size:training_batch_size+test_batch_size, :]
test_data_y = data_y[training_batch_size:training_batch_size+test_batch_size, :]

# Create and compile model
model = keras.Sequential([
    layers.Input(shape=(data_x.shape[1],)),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(2)
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train model
history = model.fit(training_data_x, training_data_y, epochs=10)

# Create prediction model
prediction_model = keras.Sequential([model, keras.layers.Softmax()])
output = np.argmax(prediction_model.predict(test_data_x), axis=1)

# Create dataframe for results
df = pd.DataFrame({
    'actual': test_data_y.ravel(),
    'predicted': output.ravel()
})

# Add columns for false negatives, true negatives, true positives, and false positives
for condition, label in zip([(1, 0), (0, 0), (1, 1), (0, 1)], 
                            ['false negative', 'true negative', 'true positive', 'false positive']):
    df[label] = (df['actual'] == condition[0]) & (df['predicted'] == condition[1])

# Plot results
plot = df.iloc[:, 2:].sum().plot(kind='bar')
plt.show()
plt.xticks(rotation=90)