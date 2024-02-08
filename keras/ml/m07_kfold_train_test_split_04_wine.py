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

#3.컴파일 훈련
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
# from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)

#2.모델
model = SVC()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC:",scores,"\n평균:",round(np.mean(scores),4))
from sklearn.metrics import accuracy_score
y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print("acc", acc)
'''
ACC: [1.         0.97222222 1.         1.         0.97142857] 
 평균: 0.9887
'''