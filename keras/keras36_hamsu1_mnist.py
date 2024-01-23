import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.models import Sequential, Model
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# ### 스케일링 1-1
# x_train = x_train /255            minmaxscaler랑 비슷
# x_test = x_test /255

### 스게일링 1-2
x_train = (x_train - 127.5)/127.5           #중위값을 맞춰주는것
x_test = (x_test - 127.5)/127.5             #standrad랑 비슷

# x_train = x_train.reshape    (60000, 28*28)
# x_test = x_test.reshape(10000,28*28)
# ### 스케일링 2-1
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)
# ### 스케일링 2-2
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test= scaler.transform(x_test)


x_train = x_train.reshape (60000, 28, 28, 1)
x_test = x_test.reshape (10000,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
input1= Input(shape=(28,28,1))
conv1 = Conv2D(10,(2,2),activation='relu')(input1)
conv2 = Conv2D(20,(3,3),activation='relu')(conv1)
conv3 = Conv2D(30,(4,4),activation='relu')(conv2)
conv4 = Conv2D(30,(4,4),activation='relu')(conv3)
conv5 = Conv2D(30,(4,4),activation='relu')(conv4)
flat1=Flatten()(conv5)
dense2= Dense(40,activation='relu')(flat1)
dense3= Dense(30,activation='relu')(dense2)
dense4= Dense(20,activation='relu')(dense3)
dense5= Dense(10,activation='relu')(dense4)
output= Dense(10,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output)


# 모델의 입력 형태를 (28, 28, 1)로 설정했지만, model.fit에서는 4D 텐서를 전달하는데, 이로 인해 모델과 데이터의 차원이 맞지 않아서 발생한 오류입니다.
# 모델의 입력 형태를 맞추기 위해 Flatten() 레이어를 추가하여 4D 텐서를 1D로 펼치고, 레이블을 원-핫 인코딩으로 변환하여 해결하였습니다.
from keras.callbacks import EarlyStopping,ModelCheckpoint
es= EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\keras25_MCP19.hdf5'
    )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2,callbacks=[es,mcp])
end_time = time.time()

result = model.evaluate(x_test, y_test)
print("loss", result[0])
print("acc", result[1])
print("걸린시간 :",round(end_time - start_time))

'''
loss 0.9844655990600586
acc 0.661899983882904
걸린시간 : 240

loss 0.04153173044323921
acc 0.9904000163078308
걸린시간 : 67

loss 0.03112885169684887
acc 0.9911999702453613
걸린시간 : 70
'''
