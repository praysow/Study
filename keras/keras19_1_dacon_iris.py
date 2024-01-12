from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#1.데이터
path= "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['species'], axis=1)
y= train_csv['species']
y_ohe= pd.get_dummies(y)




print(y_ohe)
print(y_ohe.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, train_size=0.86,
                                                    random_state=100,        #346
                                                    #stratify=y_ohe           
                                                    )

model=Sequential()
model.add(Dense(30,input_dim=4,activation='relu'))
model.add(Dense(70,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(90,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(3, activation='softmax'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=150,
                   verbose=1,
                   restore_best_weights=True
                   )
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(x_train, y_train, epochs=3000,batch_size=600, validation_split=0.3,verbose=2,
          callbacks=[es]
            )
#4.결과예측
loss=model.evaluate(x_test,y_test)
y_submit=model.predict(test_csv)
# y_submit=abs(model.predict(test_csv))
y_test = np.argmax(y_test,axis=1)
y_submit= np.argmax(y_submit,axis=1)

sampleSubmission_csv['species'] = np.round(y_submit)
print(sampleSubmission_csv)
sampleSubmission_csv.to_csv(path +"sub_3.csv", index=False)
# print("로스 :",loss)
# print("음수 개수:",sampleSubmission_csv[sampleSubmission_csv['count']<0].count())

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)

def ACC(x_train,y_train):
    return accuracy_score(y_test,np.round(y_predict))
acc = ACC(y_test,y_predict)
print("ACC :",acc)
print("로스 :",loss)


"""
ACC : 1.0
로스 : [0.47658252716064453, 1.0]   random=100 patience=100

ACC : 0.9411764705882353
로스 : [0.14077714085578918, 0.9411764740943909]    2번

ACC : 1.0
로스 : [0.04039078578352928, 1.0]   3번
"""