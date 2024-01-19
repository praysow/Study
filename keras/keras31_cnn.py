from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D




model= Sequential()
model.add(Conv2D(10,(2,2), input_shape=(10,10,1)))          #10은 다음 레이어로 전달해주는 아웃풋의 값이다. (2,2)는 나누려는 갯수(커널사이즈) (10,10,1)은 가로10개 세로10개 1장짜리 그림이다
model.add(Dense(5))
model.add(Dense(1))






