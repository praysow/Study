#첫 데이터에는 225을 초과할수 없다.
import numpy as np
import pandas as pd
from keras.datasets import  mnist
from keras.layers import Dense
(x_train, y_train),(x_test,y_test ) =mnist.load_data()
print(x_train[28])
print(np.unique(x_train,return_counts=True))
print(pd.value_counts(y_train))


# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   (10000, 28, 28) (10000,)
#다음은 리쉐이프 해주기!!
