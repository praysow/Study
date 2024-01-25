#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D,Dropout,AveragePooling2D
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder

np_path='c:/_data/_save_npy/'
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])

x_train =np.load(np_path + 'keras39_1_x_train.npy')
y_train =np.load(np_path + 'keras39_1_y_train.npy')
x_test =np.load(np_path + 'keras39_1_x_test.npy')
y_test =np.load(np_path + 'keras39_1_y_test.npy')




print(x_train)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


#2. 모델구성
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(100, 100,1)))       #input depth must be evenly divisible by filter depth: 1 vs 3
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Conv2D(20, (2, 2)))
model.add(MaxPooling2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(300))
# model.add(Dropout(0.4))
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
model.fit(x_train,y_train,epochs=1000,
          steps_per_epoch=16,           #전체데이터 나누기 batch (160/10=16)
                            #17은에러 15는 데이터 손실이다
        #batch_size=5,  # fit_generator에서는 에러, fit에서는 안먹힘,이미 batch사이즈가 위에서 들어가 있음
        verbose=2,
        # validation_data=x_test,y_test
        validation_split=0.1,
        callbacks=[es,mcp]
        )                           #vildation_split은 tensor와 numpy에서만 작동한다
end_t= time.time()

#4. 모델 평가
result = model.evaluate(x_test,y_test)
# y_submit= model.predict(xy_test)
# sample_csv=y_submit

# sample_csv.to_csv(path_train + "sample_submission_19.csv", index=False)


print("Loss:", result[0])
print("Accuracy:", result[1])
# print("걸린 시간:", round(end_t - start_t))


'''
Loss: 0.8322620391845703
Accuracy: 0.75
걸린 시간: 48
'''
