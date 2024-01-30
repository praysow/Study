#_data/kaggle/man_women 5번6번만들기#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from keras.utils import to_categorical
start_t = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)
path_train= "c:/_data/image/horse_human/train/"
# print(xy_train)
test=test_datagen.flow_from_directory(path_train,target_size=(300,300),batch_size=100,class_mode='binary')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))





np_path='c:/_data/_save_npy/'
x= np.load(np_path + 'keras39_7_x_train.npy')
y= np.load(np_path + 'keras39_7_y_train.npy')
test_horse= np.load(np_path + 'keras39_7_x_train.npy')
test_human= np.load(np_path + 'keras39_7_y_train.npy')

# y=y.values.reshape(1,-1)

# ohe = OneHotEncoder(sparse=False)
ohe = OneHotEncoder()
# y_ohe = ohe.fit_transform(y).toarray()
y=pd.get_dummies(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1,stratify=y)
x_train = x_train.reshape(-1, 300, 300)
x_test = x_test.reshape(-1, 300, 300)
# test = test.valuse.reshape(-1,300,300)

# #2. 모델구성
model = Sequential()
model.add(LSTM(30, input_shape=(300, 300)))
model.add(Dense(300))
model.add(Dense(32))
model.add(Dense(40))
model.add(Dense(2, activation='softmax'))

# model.summary()


#3. 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-2.hdf5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# start_t = time.time()
model.fit(x_train,y_train,epochs=10, batch_size=5, verbose=2,
          validation_data=(x_test, y_test),
           callbacks=[es,mcp]
        )
end_t= time.time()

#4. 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(test_horse)
y_test = np.argmax(y_test)
y_predict = np.argmax(y_submit)
# if 0.5<y_submit:
#         print('남')
# if 0.5>y_submit:
#         print('여')
y_pred= model.predict(x_test)
# y_pred= ohe.inverse_transform(y_pred)
# y_test = ohe.inverse_transform(y_test)
# f1=f1_score(y_test,y_pred, average='macro')

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))
# f1 = f1_score(y_test, y_pred, average='macro')

'''
Loss: 0.664334237575531
Accuracy: 0.675000011920929
걸린 시간: 6
'''
