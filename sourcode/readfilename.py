import os
import librosa
import numpy as np
import keras
import pandas as pd
path="C:/Users/ASUS/Desktop/Ravdess_model/record/"
predictresult = []

loaded_model = keras.models.load_model('C:/Users/ASUS/Desktop/Ravdess_model/Emotion_Voice_Detection_Model.h5')
loaded_model.summary()    
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
print(predictresult)


df1=pd.DataFrame({"predict":predictresult,"times":times}) 
df1
#P_columns=["predict","time"]
#cola=pd.DataFrame(columns=P_columns,data=predictresult)
    
        
    