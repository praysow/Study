#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder

np_path='c:/_data/_save_npy/'
x_train=np.load(np_path + 'keras39_3_x_train.npy')
y_train=np.load(np_path + 'keras39_3_y_train.npy')
# test=np.load(np_path + 'keras39_3_x_test.npy')

test_datagen = ImageDataGenerator(rescale=1./255)

path_test= "c:/_data/image/cat_and_dog/test/"
test=test_datagen.flow_from_directory(path_test,target_size=(200,200),batch_size=100,class_mode='binary',color_mode='grayscale')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다

test = np.array(test)
# print(x_train)
# print(x_train.shape)(1600, 100, 100, 1)
# print(y_train.shape)(1600,)
# print(test.shape)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=0.9,random_state=1,stratify=y_train)

x_train=x_train.reshape(-1,200,200)
x_test=x_test.reshape(-1,200,200)
#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=30,kernel_size=2, input_shape=(200, 200)))
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(32))
model.add(Dense(4))
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

#4. 모델 평가
loss = model.evaluate(x_test,y_test,verbose=0)
y_prediect = model.predict(test)
y_prediect = np.around(y_prediect.reshape(-1))
print(y_prediect.shape)
from PIL import Image
import os
sample_npy=y_prediect
path='c:/_data/kaggle/catdog/제출/'
image_path= 'C:/_data/image/cat_and_dog//test/Test'

file = os.listdir(image_path)
# y_prediect = np.around(y_prediect.reshape(-1))
# model.save(path+f"model_save//acc_{loss[1]:.6f}.h5")
y_submit = pd.DataFrame({'id': file, 'Target': y_prediect})
id_list = os.listdir(image_path)
for i in range(len(file)):
    file[i] = file[i].replace('.jpg', '')
    # 수정된 부분
    y_submit.at[i, 'id'] = file[i]  # 파일 확장자명을 '.jpg'에서 ''으로 변경

for id in id_list:
    print(id)

y_submit.to_csv(path+f"submit\\제출2.csv",index=False)
print("Loss:", loss[0])
print("Accuracy:", loss[1])
'''
Loss: 0.6651195287704468
Accuracy: 0.59375

'''