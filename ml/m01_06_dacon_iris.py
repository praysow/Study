from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

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

from sklearn.svm import LinearSVC
model = LinearSVC()

model.fit(x_train,y_train)

# 4.결과예측
result = model.score(x_test,y_test)
print("acc :", result)
y_predict = model.predict(x_test)
print(y_predict)
f1=f1_score(y_test,y_predict, average='macro')


print("f1",f1)
print("로스:", result)


"""
ACC : 1.0
로스 : [0.47658252716064453, 1.0]   random=100 patience=100

ACC : 0.9411764705882353
로스 : [0.14077714085578918, 0.9411764740943909]    2번

ACC : 1.0
로스 : [0.04039078578352928, 1.0]   3번
"""