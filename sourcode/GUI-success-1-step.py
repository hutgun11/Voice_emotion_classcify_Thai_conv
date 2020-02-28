import tkinter as tk
from tkinter import *
import keras
import numpy as np
import librosa
import pyaudio
import wave
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pandas import DataFrame

gui = Tk()

class mic():
   def __init__(self,master):
        self.master = master
    
        button_width = 10
        button_padx = "2m"    
        button_pady = "3m"
    
        self.btn1 = Button(self.master,text = 'Record',fg='white',command = self.rec)
        self.btn2 = Button(self.master,text = 'Predict',fg='white',command = self.predict)
        self.btn3 = Button(self.master,text = 'Visualize',fg='white')#,command = self.vis)
        self.btn4 = Button(self.master,text = 'Quit',fg='white',command = self.master.destroy)
        
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
              background= "pink",  
              width=button_width,  
              padx=button_padx,     
              pady=button_pady     
              )
        self.btn4.pack(side = LEFT)
        self.btn4.configure(
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
      
   def predict(self):
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
       
#   def vis(self):
#       Data1 = {'Country': ['US','CA','GER','UK','FR'],
#        'GDP_Per_Capita': [45000,42000,52000,49000,47000]
#       }
#
#       df1 = DataFrame(Data1, columns= ['Country', 'GDP_Per_Capita'])
#       df1 = df1[['Country', 'GDP_Per_Capita']].groupby('Country').sum()
#       
#       figure1 = plt.Figure(figsize=(6,5), dpi=100)
#       ax1 = figure1.add_subplot(111)
#       bar1 = FigureCanvasTkAgg(figure1, gui)
#       bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
#       df1.plot(kind='bar', legend=True, ax=ax1)
#       ax1.set_title('Country Vs. GDP Per Capita')
       
display = mic(gui)
gui.mainloop()