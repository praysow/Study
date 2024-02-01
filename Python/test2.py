from sklearn.preprocessing import OneHotEncoder
from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10, test_split=0.2)

# 전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
y_train_onehot = ohe.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = ohe.transform(y_test.reshape(-1, 1))

# 모델 생성
model = Sequential()
model.add(Embedding(input_dim=90980, output_dim=30))  # input_length 추가
model.add(Dense(10))
model.add(Dense(46, activation='softmax'))

# 콜백 및 컴파일
from keras.callbacks import EarlyStopping,ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=1000, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\\reuters.hdf5'
)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train_onehot, epochs=100, batch_size=3000, validation_split=0.3, verbose=2,
                 callbacks=[es, mcp])

# 결과 예측
result = model.evaluate(x_test, y_test_onehot)
y_pred = model.predict(x_test)
print("loss:", result[0])
print("acc:", result[1])
print("예측값:", y_pred)
