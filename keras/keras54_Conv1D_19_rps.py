#_data/kaggle/man_women 5번6번만들기#test 폴더는 사용하지 말것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder
start_t = time.time()
test_datagen = ImageDataGenerator(rescale=1./255)
# path_test="c:/_data/kaggle/man_women/test"
# print(xy_train)
# test=test_datagen.flow_from_directory(path_test,target_size=(100,100),batch_size=1600,class_mode='binary')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다
# print(np.unique(xy_train,return_counts=True))
path_test= "c:/_data/image/rps/test/"
test=test_datagen.flow_from_directory(path_test,target_size=(150,150),batch_size=200,class_mode='binary',color_mode='rgb')            #원본데이터는 최대한 건드리지 말자,원본데이터는 각각다르니 target_size를 통해서 사이즈를 동일화시킨다

test_generator = test_datagen.flow_from_directory(
    path_test,
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False  # 파일의 순서를 유지하기 위해 shuffle을 False로 설정
)

# 테스트 데이터의 전체 샘플 수 가져오기
num_samples = len(test_generator.filenames)

# 이미지를 저장할 빈 배열 생성
test = np.empty((num_samples, 150, 150, 1), dtype=np.float32)

# 배치를 반복하며 데이터를 배열에 누적
for i in range((num_samples - 1) // 100 + 1):  # num_samples가 100의 배수가 아닌 경우를 고려하여 조정
    start_idx = i * 100
    end_idx = start_idx + min(100, num_samples - start_idx)  # 마지막 배치 처리
    test[start_idx:end_idx] = test_generator.next()[0]

# NumPy 배열의 모양 출력
print("x_test의 형태:", test.shape)

test= np.array(test)

np_path='c:/_data/_save_npy/'
x=np.load(np_path + 'keras39_1_paper_train.npy')
y=np.load(np_path + 'keras39_1_rock_train.npy')
y=pd.get_dummies(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1)
x_train=x_train.reshape(-1,150,150)
x_test=x_test.reshape(-1,150,150)
test=test.reshape(-1,150,150)
#2. 모델구성
model = Sequential()
model.add(Conv1D(filters=100,kernel_size=2, input_shape=(150, 150)))
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(320))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(3, activation='softmax'))

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
y_submit= model.predict(test)
print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))


'''
Loss: 2.3304641246795654
Accuracy: 0.574999988079071

Loss: 290.3875427246094
Accuracy: 0.75
'''
