#_data/kaggle/man_women 5번6번만들기#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
start_t = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)
path_test="c:/_data/kaggle/man_women/test"
# print(xy_train)
test=test_datagen.flow_from_directory(path_test,target_size=(100,100),batch_size=1600,class_mode='binary')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))

np_path='c:/_data/_save_npy/'
x= np.load(np_path + 'keras39_5_x_train.npy')
y= np.load(np_path + 'keras39_5_y_train.npy')
# test = np.load(np_path + 'keras39_5_x_test.npy')

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1,stratify=y)


#2. 모델구성
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(100, 100,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Conv2D(20, (2, 2)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(300))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Dense(4))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

#3. 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='auto',patience=100,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-2.hdf5')
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# start_t = time.time()
model.fit(x_train,y_train,epochs=10, batch_size=5, verbose=2,
          validation_data=(x_test, y_test),
           callbacks=[es,mcp]
        )
end_t= time.time()

#4. 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(test)
# sample_csv=y_submit

# sample_csv.to_csv(path_train + "sample_submission_19.csv", index=False)

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))
