import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# 1.데이터
path= "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
y_ohe= pd.get_dummies(test_csv)
# print(y_ohe)
# print(y_ohe.shape)

x= train_csv.drop(['species'],axis=1)
y= train_csv['species']
r = int(np.random.uniform(1, 1000))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

# y_train_df = pd.DataFrame(y_train, columns=['Class_0', 'Class_1', 'Class_2'])
# y_test_df = pd.DataFrame(y_test, columns=['Class_0', 'Class_1', 'Class_2'])
# 2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=3))  # Output nodes set to the number of classes (3)
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(1, activation='softmax'))

# 3.컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3)

# 4.결과예측
loss=model.evaluate(x_test,y_test)
y_predict=model.predict([test_csv])

y_test = np.argmax(y_test,axis=1)
y_predict= np.argmax(y_predict,axis=1)
def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predict))
acc = ACC(y_test,y_predict)
print("ACC :",acc)
print("로스 :",loss)
