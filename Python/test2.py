import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
# 1.데이터
datasets = load_iris()
x = datasets.data
y = to_categorical(datasets.target)  # One-hot encoding for multi-class classification

r = int(np.random.uniform(1, 1000))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=r)

y_train_df = pd.DataFrame(y_train, columns=['Class_0', 'Class_1', 'Class_2'])
y_test_df = pd.DataFrame(y_test, columns=['Class_0', 'Class_1', 'Class_2'])
# 2.모델구성
model = Sequential()
model.add(Dense(3, input_dim=4))  # Output nodes set to the number of classes (3)
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(3, activation='softmax'))

# 3.컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3)

# 4.결과예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
def one_hot_encode_numpy(y):
    num_classes = y.shape[1]
    y_encoded = np.zeros_like(y, dtype=int)
    y_encoded[np.arange(len(y)), y.argmax(axis=1)] = 1
    return y_encoded

y_train_np = one_hot_encode_numpy(y_train)
y_test_np = one_hot_encode_numpy(y_test)

print("로스:", loss)
print("random값 :", r)


# # 원핫 인코딩 결과 출력
# print("y_train_np:\n", y_train_np)
# print("y_test_np:\n", y_test_np)