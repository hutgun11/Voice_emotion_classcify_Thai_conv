import tkinter as tk
from tkinter import *
import keras
import numpy as np
import librosa
from librosa import display
import pyaudio
import pandas as pd
import wave
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pandas import DataFrame
import time 
import pylab
from pydub import AudioSegment
import os

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
        self.btn3 = Button(self.master,text = 'Split',fg='white',command = self.split)
        self.btn4 = Button(self.master,text = 'Visualize',fg='white',command = self.vis)
        self.btn5 = Button(self.master,text = 'Quit',fg='white',command = self.master.destroy)
        
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
              background= "#000080",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn4.pack(side = LEFT)
        self.btn4.configure(
              background= "gray",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn5.pack(side = LEFT)
        self.btn5.configure(
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
   def split(self):
        audio = AudioSegment.from_wav("C:/Users/ASUS/Desktop/Ravdess_model/Test.wav") 
        path="C:/Users/ASUS/Desktop/Ravdess_model/record/"
        predictresult = []
        loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
        loaded_model.summary()   
        times=[]  
        n = len(audio) 
        counter = 1
        interval = 3 * 1000
        overlap = 1.5 * 1000
        start = 0
        end = 0
        flag = 0
        for i in range(0, 2 * n, interval): 
            if i == 0: 
                start = 0
                end = interval 
            else: 
                start = end - overlap 
                end = start + interval  
            if end >= n: 
                end = n 
                flag = 1
            chunk = audio[start:end] 
            filename = 'chunk'+str(counter)+'.wav'
            chunk.export("C:/Users/ASUS/Desktop/Ravdess_model/record/"+filename, format ="wav") 
            print("Processing chunk "+str(counter)+". Start = "
                                +str(start)+" end = "+str(end)) 
            times.append(end/1000)
            counter = counter + 1
            
        for subdir, dirs, filesname in os.walk(path):
          for file in filesname:
              try:
                data, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0) 
                x = np.expand_dims(mfccs, axis=2)
                x = np.expand_dims(x, axis=0)
            
                predict=loaded_model.predict_classes(x)
                predictresult.append(predict)
              except ValueError:
                continue    
            #print(predictresult)
        for i in range(0,len(predictresult)):
                predictresult[i]=int(predictresult[i])
        
#        librosa.display.waveplot(data, sr=sampling_rate)
        plt.plot(times,predictresult)    
        librosa.display.waveplot(data, sr=sampling_rate)
        plt.subplot(111)
        plt.show() 
        df1=pd.DataFrame({"predict":predictresult,"times":times}) 
#        librosa.display.waveplot(data, sr=sampling_rate)
#        df1.plot(x='times',y='predict',c='red')
#        plt.plot(times,predictresult)    
#        plt.subplot(111)
#        plt.show()  
        #พลอต2กราฟ
         
        
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
       #------------wave------------#
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(data, sr=sampling_rate)
        plt.subplot(111)
        plt.show()
        #------------wave------------#
    
        
display = mic(gui)
gui.title('Sound Check')
gui.resizable(height = None, width = None)
gui.mainloop()

