from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,AveragePooling2D,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cat= "c:\_data\kaggle\catdog\Cat\\"
cat_csv =plt.imread(cat+"0.jpg")
# dog_csv= pd.read_csv(path+"dog.jpg")
# test_csv= pd.read_csv(path+"test.jpg")


# print(cat_csv.shape)
# plt.imshow(cat_csv)
# plt.show()

import os
from PIL import Image
cat =  "c:\_data\kaggle\catdog\Cat\\"
cat_list = os.listdir(cat)

