import tkinter as tk
from tkinter import *
import keras
import numpy as np
import librosa
from librosa import display
import pyaudio
import wave
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pandas import DataFrame
import time 
import pylab

gui = Tk()
data, sampling_rate = librosa.load('C:/Users/ASUS/Desktop/Ravdess_model/Test.wav')
CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "C:/Users/ASUS/Desktop/Ravdess_model/Test.wav"
class mic():
   def __init__(self,master):
        self.master = master
        self.label = tk.Label(text="")
        
        button_width = 10
        button_padx = "2m"    
        button_pady = "3m"
        self.mlabel = Label(text = 'How Do You Feel??',font=("Courier 12 bold"))
        self.btn1 = Button(self.master,text = 'Record',fg='white',command = self.rec)
        self.btn2 = Button(self.master,text = 'Predict',fg='white',command = self.predict)
        self.btn3 = Button(self.master,text = 'Visualize',fg='white',command = self.vis)
        self.btn4 = Button(self.master,text = 'Quit',fg='white',command = self.master.destroy)
        
        self.label.pack()
        self.update_clock()
        self.mlabel.pack()
        
        self.btn1.pack(side = LEFT)
        self.btn1.configure( 
              background= "#009966",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn2.pack(side = LEFT)
        self.btn2.configure(
              background= "#0066FF",
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn3.pack(side = LEFT)
        self.btn3.configure(
              background= "gray",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn4.pack(side = LEFT)
        self.btn4.configure(
              background= "#CD5C5C",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        
   def rec(self):
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
       self.mlabel.config(text = '* done recording')
       stream.stop_stream()
       stream.close()
       p.terminate()
       wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
       wf.setnchannels(CHANNELS)
       wf.setsampwidth(p.get_sample_size(FORMAT))
       wf.setframerate(RATE)
       wf.writeframes(b''.join(frames))
       wf.close()
      
   def predict(self):
       data, sampling_rate = librosa.load('C:/Users/ASUS/Desktop/Ravdess_model/Test.wav')
       mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
       x = np.expand_dims(mfccs, axis=2)
       x = np.expand_dims(x, axis=0)
       loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
       predictions = loaded_model.predict_classes(x)
       
       if predictions == 1 :
           result ="Your feel is neutral"
       elif predictions == 4 :
           result = "Your feel is angry"    
       print(result)
       #change text in Lavbel-----------
       self.mlabel.config(text = result)
       #-------------------------------
   def update_clock(self):
        now = time.strftime("%H:%M:%S")
        self.label.configure(text=now)
        self.master.after(1000, self.update_clock)
        
   def vis(self):
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(data, sr=sampling_rate)
        plt.subplot(111)
        plt.show()
       
display = mic(gui)
gui.title('Sound Check')
gui.resizable(height = None, width = None)
gui.mainloop()