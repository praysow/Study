import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
datasets=load_wine()
x= datasets.data
y= datasets.target
# print(x.shape,y.shape)  (178, 13) (178,)
# print(np.unique(y, return_counts=True)) (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y))
# print(y)

x = x[:-30]
y = y[:-30]
# print(y)
# print(np.unique(y, return_counts=True))
print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=3,stratify=y)   #stratify=y의 비율대로 잘라라

from imblearn.over_sampling import SMOTE
import sklearn as sk
# print("sk",sk.__version__)

smote= SMOTE(random_state=1)
x_train,y_train = smote.fit_resample(x_train,y_train)
print(x_train.shape,y_train.shape)
# print(pd.value_counts(y_train))
'''
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization

model=Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(3,activation='softmax'))

from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\대출_46.hdf5'
    )
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='accuracy')       #원핫을 하지않으려면 sparse를 사용한다
model.fit(x_train,y_train,epochs=1000,batch_size=10, validation_split=0.1,verbose=3,callbacks=[es,mcp])

results = model.evaluate(x_test, y_test)
y_pred= model.predict(x_test)
y_pred= np.argmax(y_pred, axis=1)
f1=f1_score(y_test,y_pred, average='micro')

print("f1",f1)
print("loss:",results[0])
print("acc:",results[1])

# print(x_test)   #원핫no
# print(y_pred)   #원핫yes
'''
'''
f1 0.9474747474747475
loss: 0.11577361077070236
acc: 0.9333333373069763
'''
