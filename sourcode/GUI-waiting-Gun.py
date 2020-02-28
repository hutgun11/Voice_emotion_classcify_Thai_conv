import tkinter as tk
from tkinter import *
import keras
import numpy as np
import librosa
import pyaudio
import wave

gui = Tk()

class mic():
   def __init__(self,master):
        self.master = master
    
        button_width = 10
        button_padx = "2m"    
        button_pady = "3m"
    
        self.btn1 = Button(self.master,text = 'Record',fg='white',command = self.rec)
        self.btn2 = Button(self.master,text = 'Predict',fg='white')
        self.btn3 = Button(self.master,text = 'Quit',fg='white',command = self.master.destroy)
        
        self.btn1.pack(side = LEFT)
        self.btn1.configure( 
              background= "green",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn2.pack(side = LEFT)
        self.btn2.configure(
              background= "blue",
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn3.pack(side = LEFT)
        self.btn3.configure(
              background= "red",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        
   def rec(self):
       CHUNK = 1024 
       FORMAT = pyaudio.paInt16 #paInt8
       CHANNELS = 2 
       RATE = 44100 #sample rate
       RECORD_SECONDS = 4
       WAVE_OUTPUT_FILENAME = "C:/Users/moskr/Desktop/Ravdess_model/Test.wav"
        
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
       data, sampling_rate = librosa.load('C:/Users/moskr/Desktop/Ravdess_model/Test.wav')
       mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
       x = np.expand_dims(mfccs, axis=2)
       x = np.expand_dims(x, axis=0)

        
display = mic(gui)
gui.mainloop()