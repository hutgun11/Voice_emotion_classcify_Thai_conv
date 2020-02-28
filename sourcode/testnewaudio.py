import librosa
from librosa import display
import matplotlib.pyplot as plt
import os
import pandas as pd

import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import keras

import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
path = 'D:/sound/tess/total tess'
lst = []

start_time = time.time()

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1 
        arr = mfccs, file
        lst.append(arr)
      # If the file is not valid, skip it
      except ValueError:
        continue

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)

X.shape, y.shape


X_name = 'Xtest.joblib'
y_name = 'ytest.joblib'
save_dir = 'C:/Users/ASUS/Desktop/Ravdess_model'

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))


Xtest = joblib.load('C:/Users/ASUS/Desktop/Ravdess_model/Xtest.joblib')
ytest = joblib.load('C:/Users/ASUS/Desktop/Ravdess_model/ytest.joblib')
x_testcnn = np.expand_dims(Xtest, axis=2)
x_testcnn.shape
# Save model and weights

loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
loaded_model.summary()
loss, acc = loaded_model.evaluate(x_testcnn, ytest)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))