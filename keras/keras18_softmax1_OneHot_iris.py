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
y = datasets.target
y_ohe1 = to_categorical(datasets.target)  # One-hot encoding for multi-class classification
# print(datasets.DESCR)
# print(datasets.feature_names)
# print("x",x.shape)      #(150,4)
# print("y",y.shape)      #(150, )
# print(np.unique(y,return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y)) 
# 0    50
# 1    50
# 2    50
#판다스
# y_ohe2 = pd.get_dummies(y)
# print(y_ohe2.shape)
# print(y_ohe2)

# #사이킷런
# y= y.reshape(-1,1)

# OHE = OneHotEncoder(sparse=False)#디폴트는 True
# OHE = OneHotEncoder()
# # OHE.fit(y)                     #import>fit>transform
# # y_ohe3 = OHE.transform(y)      #fit으로 저장을하고 transform으로 바꾼다
# # y_ohe3 = OHE.fit_transform(y)    # 간단하것으로 바꾸는 것
# y_ohe3 = OHE.fit_transform(y).toarray()


r = int(np.random.uniform(1, 1000))
x_train, x_test, y_train, y_test = train_test_split(x, y_ohe1, train_size=0.8,
                                                    random_state=r,         #850:acc=1
                                                    stratify=y              #stratify는 분류에서만 사용
                                                    )

# y_train_df = pd.DataFrame(y_train, columns=['Class_0', 'Class_1', 'Class_2'])
# y_test_df = pd.DataFrame(y_test, columns=['Class_0', 'Class_1', 'Class_2'])

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
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy',
                   mode='max',
                   patience=100,
                   verbose=1,
                   restore_best_weights=True
                   )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=30, validation_split=0.3,callbacks=[es])

# 4.결과예측

result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test,axis=1)                   #열의 값을 맞추기 위해서는 argmax를 통과시킨다
y_predict= np.argmax(y_predict,axis=1)              #소수점으로 나누어져있는것을 가독성있게 0에서1사이값으로 맞춰준다

print(y_test)
print(y_predict)
print(y_test.shape,y_predict.shape)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict,y_test)
print("accuracy_score :", acc)



# def one_hot_encode_numpy(y):
#     num_classes = y.shape[1]
#     y_encoded = np.zeros_like(y, dtype=int)
#     y_encoded[np.arange(len(y)), y.argmax(axis=1)] = 1
#     return y_encoded

# y_train_np = one_hot_encode_numpy(y_train)
# y_test_np = one_hot_encode_numpy(y_test)

print("로스 :", result[0])
print("acc :",result[1])
print("random값 :", r)

# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
# font_path = datasets
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)


# plt.figure(figsize=(65,6))
# # plt.scatter(hist.history['loss'])
# plt.plot(hist.history['loss'],c='red', label='loss',marker='.')
# plt.plot(hist.history['val_loss'],c='blue', label='loss',marker='.')
# # plt.plot(hist.history['r2'],c='pink', label='loss',marker='.')
# plt.legend(loc='upper right')
# plt.title('데이콘 로스')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


