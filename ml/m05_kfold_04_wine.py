import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Conv1D,Flatten
from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#1.데이터
datasets= load_wine()
x= datasets.data
y= datasets.target

# # 사이킷런

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=383,        
                                                    # stratify=y_ohe1            
                                                    )

from sklearn.model_selection import KFold, cross_val_score

n_split = 5
# kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = LinearDiscriminantAnalysis()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
'''
ACC: [1.         0.97222222 1.         1.         0.97142857] 
 평균: 0.9887
'''