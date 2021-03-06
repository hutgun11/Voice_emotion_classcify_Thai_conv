# Import necessary libraries 
from pydub import AudioSegment 
import os
import librosa
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
# Input audio file to be sliced 
audio = AudioSegment.from_wav("C:/Users/ASUS/Desktop/Ravdess_model/Test.wav") 
 
path="C:/Users/ASUS/Desktop/Ravdess_model/record/"
predictresult = []

loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
loaded_model.summary()   
times=[]  
 
''' 
Step #1 - Slicing the audio file into smaller chunks. 
'''
# Length of the audiofile in milliseconds 
n = len(audio) 
  
# Variable to count the number of sliced chunks 
counter = 1
  
# Text file to write the recognized audio 
#fh = open("recognized.txt", "w+") 
  
# Interval length at which to slice the audio file. 
# If length is 22 seconds, and interval is 5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 5 - 10 seconds 
# chunk3 : 10 - 15 seconds 
# chunk4 : 15 - 20 seconds 
# chunk5 : 20 - 22 seconds 
interval = 3 * 1000
  
# Length of audio to overlap.  
# If length is 22 seconds, and interval is 5 seconds, 
# With overlap as 1.5 seconds, 
# The chunks created will be: 
# chunk1 : 0 - 5 seconds 
# chunk2 : 3.5 - 8.5 seconds 
# chunk3 : 7 - 12 seconds 
# chunk4 : 10.5 - 15.5 seconds 
# chunk5 : 14 - 19.5 seconds 
# chunk6 : 18 - 22 seconds 
overlap = 1.5 * 1000
  
# Initialize start and end seconds to 0 
start = 0
end = 0
  
# Flag to keep track of end of file. 
# When audio reaches its end, flag is set to 1 and we break 
flag = 0
  
# Iterate from 0 to end of the file, 
# with increment = interval 
for i in range(0, 2 * n, interval): 
      
    # During first iteration, 
    # start is 0, end is the interval 
    if i == 0: 
        start = 0
        end = interval 
  
    # All other iterations, 
    # start is the previous end - overlap 
    # end becomes end + interval 
    else: 
        start = end - overlap 
        end = start + interval  
  
    # When end becomes greater than the file length, 
    # end is set to the file length 
    # flag is set to 1 to indicate break. 
    if end >= n: 
        end = n 
        flag = 1
  
    # Storing audio file from the defined start to end 
    chunk = audio[start:end] 
  
    # Filename / Path to store the sliced audio 
    filename = 'chunk'+str(counter)+'.wav'
  
    # Store the sliced audio file to the defined path 
    chunk.export("C:/Users/ASUS/Desktop/Ravdess_model/record/"+filename, format ="wav") 
    # Print information about the current chunk 
    print("Processing chunk "+str(counter)+". Start = "
                        +str(start)+" end = "+str(end)) 
    times.append(end/1000)
    
    # Increment counter for the next chunk 
    counter = counter + 1
      
    # Slicing of the audio file is done. 
    # Skip the below steps if there is some other usage 
    # for the sliced audio files. 
for subdir, dirs, filesname in os.walk(path):
  for file in filesname:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        data, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T,axis=0) 
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        
        predict=loaded_model.predict_classes(x)
        predictresult.append(predict)
      # If the file is not valid, skip it
      except ValueError:
        continue    
#print(predictresult)
for i in range(0,len(predictresult)):
    predictresult[i]=int(predictresult[i])
df1=pd.DataFrame({"predict":predictresult,"times":times}) 
df1
ax1 = df1.plot.scatter(x='times',y='predict',c='DarkBlue')