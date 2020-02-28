import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras import regularizers
# Load training data
data = pd.read_csv('D:/DOwnload/Hackthon/Trainset_Homedottech_Hackathon.csv', encoding='utf8')
# Load test data
datatest  = pd.read_csv('D:/DOwnload/Hackthon/Testset_Homedottech_Hackathon.csv', encoding='utf8') 
dataset = data.values
dataset
