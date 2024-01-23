import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import OneHotEncoder

train_datagen = ImageDataGenerator(
    rescale=1.255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.0,
    # shear_range=0.9,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

path_train = "c:/_data/image/brain/train"
path_test = "c:/_data/image/brain/test"
xy_train = train_datagen.flow_from_directory(path_train, target_size=(200, 200), batch_size=10, class_mode='binary')

xy_test = test_datagen.flow_from_directory(path_test, target_size=(200, 200), batch_size=10, class_mode='binary')

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

# 데이터를 4D로 reshape
x_train = x_train.reshape(x_train.shape[0], 200, 200, 3)
x_test = x_test.reshape(x_test.shape[0], 200, 200, 3)

#2. 모델구성
model = Sequential()
model.add(Conv2D(5, (5, 5), input_shape=(200, 200, 3)))
model.add(Conv2D(10, (6, 6)))
model.add(Conv2D(15, (7, 7)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))

#3. 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='auto', patience=100, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-1.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
start_t = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=2,
          validation_split=0.1,
          callbacks=[es, mcp]
          )
end_t = time.time()

#4. 모델 평가
result = model.evaluate(x_test, y_test)
y_submit = model.predict(x_test)

print("Loss:", result[0])
print("Accuracy:", result[1])
print("걸린 시간:", round(end_t - start_t))
