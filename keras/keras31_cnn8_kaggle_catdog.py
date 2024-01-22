from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,AveragePooling2D,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

path = "c:/_data/kaggle/catdog/"
cat_csv = pd.read_csv(path +"cat")
dog_csv= pd.read_csv(path+"dog")
test_csv= pd.read_csv(path+"test")


