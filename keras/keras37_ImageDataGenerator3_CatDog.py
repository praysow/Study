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
start_t = time.time()
train_dategen=ImageDataGenerator(
     rescale=1.255,                      #앞에 1.에서 .은 부동소수점 계산하겠다는 것이다
     # horizontal_flip=True,           #수평 뒤집기
     # vertical_flip=True,             #수직 뒤집기
     # width_shift_range=0.1,           #평행이동 0.1=10%이동
     # height_shift_range=0.1,          #
     # rotation_range=5,               #정해진 각도 만큼 이미지를 회전 (5도 회전)
     # zoom_range=1.0,                 #1.2배 확대혹은 축소
     # shear_range=0.9,                 #좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
     # fill_mode='nearest'
     )

batch1=100
batch2=100

test_datagen = ImageDataGenerator(rescale=1./255)

path_train= "c:/_data/image/cat_and_dog/train/"
path_test= "c:/_data/image/cat_and_dog/test/"
xy_train=train_dategen.flow_from_directory(path_train,target_size=(batch1,batch2),batch_size=1600,class_mode='binary')         #batch_size을 최대로 하면 x데이터를 통으로 가져올수있다


# print(xy_train)
xy_test=test_datagen.flow_from_directory(path_test,target_size=(batch1,batch2),batch_size=1600,class_mode='binary')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))


x = xy_train[0][0]                 #batch가 100일때 100의 0번째이다
y = xy_train[0][1]                 #batch가 100일때 100의 1번째이다
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
# print(xy_train[0][0].shape,xy_train[0][1].shape)  #(800, 150, 150, 3) (800,)



# x_train,x_test,y_train,y_test=train_test_split(xy_train[0][0],xy_train[0][1],train_size=0.8,random_state=1,stratify=y)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1,stratify=y)

# print(x_train.shape,y_train.shape)  #(800, 100, 100, 3) (800,)
# print(x_test.shape,y_test.shape)    #(200, 100, 100, 3) (200,)

# print(x_train.next())
# end = time.time()

# print("time",end-start)


#2. 모델구성
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(batch1, batch2,3)))
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
model.fit(x_train,y_train,epochs=1000, batch_size=5, verbose=2,
          validation_data=(x_test, y_test),
           callbacks=[es,mcp]
        )
end_t= time.time()
from PIL import Image
import os
#4. 모델 평가
result = model.evaluate(x_test, y_test)
y_submit= model.predict(xy_test)
sample_npy=y_submit
path='c:/_data/kaggle/catdog/'
submission_df = pd.DataFrame(columns=['ID', 'Target'])  # 'ID' 열로 수정

for i in range(len(y_submit)):
    img_array = (y_submit[i, 0] * 255).astype(np.uint8)  # 이미지의 형태로 변환
    img_array = np.squeeze(img_array)  # 차원을 축소하여 2D로 만듦
    img_array = np.expand_dims(img_array, axis=-1)  # 차원을 다시 확장하여 (100, 100, 1)로 만듦
    img = Image.fromarray(img_array, mode='L')  # 이미지 생성
    img_path = path + f"cat_dog_{i+1}.jpg"
    img.save(img_path)

    # 제출용 데이터프레임에 정보 추가
    img_name = os.path.basename(img_path)  # 파일 경로에서 파일 이름만 추출
    img_num = int(img_name.split('_')[2].split('.')[0])  # 파일 이름에서 숫자 추출
    submission_df = pd.concat([submission_df, pd.DataFrame({'ID': [img_num], 'Target': [y_submit[i, 0]]})], ignore_index=True)

# CSV 파일로 제출용 데이터프레임 저장
submission_df.to_csv(path + "submission.csv", index=False)

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))


'''
Loss: 0.8322620391845703
Accuracy: 0.75
걸린 시간: 48
'''
