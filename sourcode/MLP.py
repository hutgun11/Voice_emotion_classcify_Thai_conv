import librosa
from librosa import display
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import keras
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import validation_curve




data, sampling_rate = librosa.load('C:/Users/ASUS/Desktop/sound/03-01-02-02-04-09-02.wav')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.title("Neutral")
data, sampling_rate = librosa.load('C:/Users/ASUS/Desktop/sound/03-01-05-02-04-09-10.wav')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
plt.title("Angry")
print(data)
path = 'C:/Users/ASUS/Desktop/sound'
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
#print(lst)
print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
X, y = zip(*lst)
X = np.asarray(X)
y = np.asarray(y)
print(X)
print(y)
X.shape, y.shape


X_name = 'X.joblib'
y_name = 'y.joblib'
save_dir = 'C:/Users/ASUS/Desktop/Ravdess_model'

savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))


X = joblib.load('C:/Users/ASUS/Desktop/Ravdess_model/X.joblib')
y = joblib.load('C:/Users/ASUS/Desktop/Ravdess_model/y.joblib')
#แบ่งข้อมูลtrain,test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#สร้างdetree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
#สร้าง rforest
rforest = RandomForestClassifier(criterion="gini", max_depth=10, max_features="log2", 
                                 max_leaf_nodes = 100, min_samples_leaf = 3, min_samples_split = 20, 
                                 n_estimators= 22000, random_state= 5)
rforest.fit(X_train, y_train)
predictions = rforest.predict(X_test)
print(predictions)
print(classification_report(y_test,predictions))
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)
x_traincnn.shape, x_testcnn.shape
## Create range of values for parameter
#param_range = np.arange(1, 250, 2)
#
## Calculate accuracy on training and test set using range of parameter values
#train_scores, test_scores = validation_curve(rforest, 
#                                             X, 
#                                             y, 
#                                             param_name="n_estimators", 
#                                             param_range=param_range,
#                                             cv=3, 
#                                             scoring="accuracy", 
#                                             n_jobs=-1)
#
##predictions = rforest.predict(X_test)
##print(classification_report(y_test,predictions))
## Calculate mean and standard deviation for training set scores
#train_mean = np.mean(train_scores, axis=1)
#train_std = np.std(train_scores, axis=1)
#
## Calculate mean and standard deviation for test set scores
#test_mean = np.mean(test_scores, axis=1)
#test_std = np.std(test_scores, axis=1)
#
## Plot mean accuracy scores for training and test sets
#plt.plot(param_range, train_mean, label="Training score", color="black")
#plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")
#
## Plot accurancy bands for training and test sets
#plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
#plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")
#
## Create plot
#plt.title("Validation Curve With Random Forest")
#plt.xlabel("Number Of Trees")
#plt.ylabel("Accuracy Score")
#plt.tight_layout()
#plt.legend(loc="best")
#plt.show()
#
#
#
#
#x_traincnn = np.expand_dims(X_train, axis=2)
#x_testcnn = np.expand_dims(X_test, axis=2)
#x_traincnn.shape, x_testcnn.shape

#plt.figure(figsize=(12, 8))
#plt.title('Random Forest')
#plt.plot(y_test, 'b-', color="black")
#plt.plot(predictions, 'b-', color="blue")
#plt.legend(loc='upper right')
#plt.xlabel('Estimators')
#plt.ylabel('Mean Squared Error')
#plt.show()
#MLP NN
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


## สร้างโมเดล CNN
model = Sequential()

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)#ปกคิใช้softmax
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=1000, validation_data=(x_testcnn, y_test))
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
predictions = model.predict_classes(x_testcnn)
predictions
y_test
new_Ytest = y_test.astype(int)
new_Ytest
report = classification_report(new_Ytest, predictions)
print(report)
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'C:/Users/ASUS/Desktop/Ravdess_model'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
loaded_model.summary()
loss, acc = loaded_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))