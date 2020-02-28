import tkinter as tk
from tkinter import *
import keras
import numpy as np
import librosa
import pyaudio
import wave
import csv
from sklearn.externals import joblib
CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "C:/Users/ASUS/Desktop/Ravdess_model/Test.wav"
        
p = pyaudio.PyAudio()
       
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer
        
print("* recording")
        
frames = []
        
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel
print("* done recording")
        
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
data, sampling_rate = librosa.load('C:/Users/ASUS/Desktop/Ravdess_model/Test.wav')
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
x = np.expand_dims(mfccs, axis=2)
x = np.expand_dims(x, axis=0)
loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
predictions = loaded_model.predict_classes(x)
if predictions == 1 :
    result ="neutral"
elif predictions == 4 :
    result = "angry"    

print(result)
#with open('C:/Users/ASUS/Desktop/Ravdess_model/sound.csv','w',newline='')as f:
#    thewriter = csv.writer(f)
#    thewriter.writerow([x])