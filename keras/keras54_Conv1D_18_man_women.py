import time
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split

# 데이터 로딩 및 전처리
test_datagen = ImageDataGenerator(rescale=1./255)
path_test = "c:/_data/kaggle/man_women/test"
test_generator = test_datagen.flow_from_directory(
    path_test,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)
num_samples = len(test_generator.filenames)
test = np.empty((num_samples, 100, 100, 3), dtype=np.float32)
for i in range((num_samples - 1) // 100 + 1):
    start_idx = i * 100
    end_idx = start_idx + min(100, num_samples - start_idx)
    test[start_idx:end_idx] = test_generator.next()[0]
test = test.reshape(-1, 100, 100 * 3)

# 모델 구성
model = Sequential()
model.add(Conv1D(filters=30, kernel_size=2, input_shape=(100, 100 * 3)))  # 수정된 부분
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(32))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))

# 데이터 분할
np_path = 'c:/_data/_save_npy/'
x = np.load(np_path + 'keras39_5_x_train.npy')
y = np.load(np_path + 'keras39_5_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1, stratify=y)

# 이미지 데이터의 shape 조정
x_train = x_train.reshape(-1, 100, 100 * 3)
x_test = x_test.reshape(-1, 100, 100 * 3)

# 모델 컴파일 및 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=5, verbose=2,
          validation_data=(x_test, y_test))

# 모델 평가
result = model.evaluate(x_test, y_test)
print("Loss:", result[0])
print("Accuracy:", result[1])

# 테스트 데이터 예측
y_submit = model.predict(test)

# 예측 결과 출력
for pred in y_submit:
    if pred > 0.5:
        print('남')
    else:
        print('여')

'''
Loss: 13.843233108520508
Accuracy: 0.550000011920929
남
'''

