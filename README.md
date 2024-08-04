# EM_Net: Efficient Emotion Interpretation through Neural Networks


This project aims to detect emotions from audio files using a neural network. It leverages the RAVDESS dataset for training and testing. The primary features extracted from audio files are MFCC, chroma, and mel spectrogram. The model used is a Multi-Layer Perceptron (MLP) classifier from the Scikit-learn library.

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Usage](#usage)
    - [1. Upload Dataset](#1-upload-dataset)
    - [2. Extract Features](#2-extract-features)
    - [3. Train the Model](#3-train-the-model)
    - [4. Evaluate the Model](#4-evaluate-the-model)
  - [Results](#results)
  - [Model Saving and Loading](#model-saving-and-loading)
  - [References](#references)

## Dataset

The dataset used in this project is the RAVDESS dataset, which contains audio recordings of actors expressing various emotions. Each audio file is labeled with the corresponding emotion.

## Requirements

- Python 3.6+
- Google Colab (recommended for running the code)
- The following Python libraries:
  - librosa
  - soundfile
  - NumPy
  - matplotlib
  - sklearn
  - google.colab

## Setup

1. Open Google Colab.
2. Mount your Google Drive to save and load the model.
3. Install the required libraries if they are not already installed.

## Usage

### 1. Upload Dataset

Upload your dataset from your local machine to Google Colab using the following code snippet:

```python
from google.colab import files
uploaded = files.upload()
```

Select the files from your local machine to upload them.

### 2. Extract Features

Define a function to extract features from the audio files:

```python
import soundfile
import numpy as np
import librosa

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result
```

### 3. Train the Model

Load the data, split it into training and testing sets, and train the model:

```python
from sklearn.model_selection import train_test_split

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def load_data(test_size=0.2):
    x, y = [], []
    for file_name in uploaded.keys():
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file_name, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train, x_test, y_train, y_test = load_data(test_size=0.25)
```

Initialize and train the model:

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train, y_train)
```

### 4. Evaluate the Model

Evaluate the model’s performance on the test set:

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Test accuracy: {:.2f}%".format(accuracy * 100))
```

## Results

The results of the model are printed as accuracy scores for both training and testing datasets. You can also plot the loss curve to visualize the model's training process:

```python
import matplotlib.pyplot as plt

loss_values = model.loss_curve_
plt.plot(loss_values)
plt.show()
```

## Model Saving and Loading

Save the trained model for future use:

```python
import pickle

pickle.dump(model, open('emotion_classification-model.pkl', 'wb'))
```

Load the saved model when needed:

```python
model = pickle.load(open('emotion_classification-model.pkl', 'rb'))
```

## References

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Google Colab](https://colab.research.google.com/)

This README file provides a comprehensive guide to understanding, setting up, and running the emotion detection project using neural networks.
