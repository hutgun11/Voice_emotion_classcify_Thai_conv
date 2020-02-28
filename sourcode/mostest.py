import keras
import numpy as np
import librosa
import pyaudio
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import tkinter as tk
from tkinter import *

%matplotlib tk

class livePredictions:
    def __init__(self):
        self.gui = tk.Tk()
        self.gui.title('Sound Check')

        self.canvas = tk.Canvas(self.gui, width='450',height='450')
        self.canvas.configure(background='pink')
        #create btn
        self.btn1 = tk.Button(self.gui,text='visualize',command = self.wow)
        self.btn2 = tk.Button(self.gui,text='Quit',command=self.gui.destroy)
        self.btn3 = tk.Button(self.gui,text = 'Predict',command= self.predict)
        
        self.canvas.pack()
        self.btn1.pack()
        self.btn2.pack()
        self.btn3.pack()

        self.gui.mainloop()
    def wow(self):
        # constants
        CHUNK = 1024 * 2             # samples per frame
        FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
        CHANNELS = 1                 # single channel for microphone
        RATE = 44100                 # samples per second
        RECORD_SECONDS = 4
        WAVE_OUTPUT_FILENAME = "Test.wav"
        # create matplotlib figure and axes
        fig, ax = plt.subplots(1, figsize=(15, 7))
        
        # pyaudio class instance
        p = pyaudio.PyAudio()
        
        # stream object to get data from microphone
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK
        )
        
        # variable for plotting
        x = np.arange(0, 2 * CHUNK, 2)
        
        # create a line object with random data
        line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)
        
        # basic formatting for the axes
        ax.set_title('AUDIO WAVEFORM')
        ax.set_xlabel('samples')
        ax.set_ylabel('volume')
        ax.set_ylim(0, 255)
        ax.set_xlim(0, 2 * CHUNK)
        plt.setp(ax, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])
        
        # show the plot
        plt.show(block=False)
        
        print('stream started')
        
        # for measuring frame rate
        frame_count = 0
        start_time = time.time()
        
        while True:
            
            # binary data
            data = stream.read(CHUNK)  
            
            # convert data to integers, make np array, then offset it by 127
            data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
            
            # create np array and offset by 128
            data_np = np.array(data_int, dtype='b')[::2] + 128
            
            line.set_ydata(data_np)
            
            # update figure canvas
            try:
                fig.canvas.draw()
                fig.canvas.flush_events()
                frame_count += 1
                
            except TclError:
                
                # calculate average frame rate
                frame_rate = frame_count / (time.time() - start_time)
                
                print('stream stopped')
                print('average frame rate = {:.0f} FPS'.format(frame_rate))
                break
        
    def load(self, path, file):

        self.path = path
        self.file = file

    def load_model(self):
        '''
        I am here to load you model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        '''
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        '''
        I am here to process the files and create your features.
        '''
        
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        
        predictions
        print(predictions)
        print( "Prediction is", " ", self.convertclasstoemotion(predictions))
      
        
   
        
    def convertclasstoemotion(self, pred):
        '''
        I am here to convert the predictions (int) into human readable strings.
        '''
        self.pred  = pred

        if pred == 1:
            pred = "neutral"
            return pred
        elif pred == 4:
            pred = "angry"
            return pred
        
    def predict(self):
# Here you can replace path and file with the path of your model and of the file from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.

        CHUNK = 1024 
        FORMAT = pyaudio.paInt16 #paInt8
        CHANNELS = 2 
        RATE = 44100 #sample rate
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = "Test.wav"
        
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
        
        wavefile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wavefile.setnchannels(CHANNELS)
        wavefile.setsampwidth(audio.get_sample_size(FORMAT))
        wavefile.setframerate(RATE)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()
        pred = livePredictions(path='C:/Users/moskr/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5',
                               file= WAVE_OUTPUT_FILENAME)
                               #file='C:/Users/ASUS/Desktop/sound/03-01-02-02-01-05-02.wav')
        
        pred.load_model()
        pred.makepredictions()
        
        
display = livePredictions()