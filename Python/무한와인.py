# test 폴더는 사용하지 말 것
import time
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout, AveragePooling2D
from sklearn.model_selection import train_test_split
import os

np_path = 'c:/_data/_save_npy/'
x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')

test_datagen = ImageDataGenerator(rescale=1./255)

path_test = "c:/_data/image/cat_and_dog/test/"
test = test_datagen.flow_from_directory(path_test, target_size=(100, 100), batch_size=1600, class_mode='binary', color_mode='grayscale')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, random_state=1, stratify=y_train)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(100, 100, 1)))
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

# 3. 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='auto', patience=100, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath='../_data/_save/MCP/keras31-2.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=5, verbose=2,
          validation_data=(x_test, y_test),
          callbacks=[es, mcp]
          )

# 4. 모델 평가
loss = model.evaluate(x_test, y_test, verbose=0)
y_prediect = model.predict(test)
y_prediect = np.around(y_prediect.reshape(-1))
print(y_prediect.shape)

path = 'c:/_data/kaggle/catdog/제출/'
y_prediect = np.around(y_prediect.reshape(-1))
model.save(path + f"model_save\\acc_{loss[1]:.6f}.h5")

y_submit = pd.DataFrame({'id': range(5000), 'Target': y_prediect})

forder_dir = path + 'submit'
id_list = os.listdir(forder_dir)
for i, id in enumerate(id_list):
    if id.startswith('acc_'):
        id_list[i] = int(id.split('_')[1].split('.')[0])

for id in id_list:
    print(id)

y_submit.to_csv(path + f"submit\\제출1.csv", index=False)

print("Loss:", loss[0])
print("Accuracy:", loss[1])
