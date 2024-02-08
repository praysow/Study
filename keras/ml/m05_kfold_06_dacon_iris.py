from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#1.데이터
path= "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['species'], axis=1)
y= train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=100,        #346
                                                    #stratify=y_ohe           
                                                    )
from sklearn.model_selection import KFold, cross_val_score

n_split = 5
# kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델
model = QuadraticDiscriminantAnalysis()   #소프트벡터머신 클래스파이어
#3.훈련
scores = cross_val_score(model,x,y,cv=kfold)

print("ACC:",scores,"\n 평균:",round(np.mean(scores),4))
"""
ACC: [0.95833333 0.95833333 1.         1.         0.95833333]
 평균: 0.975
"""