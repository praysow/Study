#스케일링후 교육용 분류 만들기
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,x_test.shape)   #(60000, 28, 28) (10000, 28, 28)

# x=np.append(x_train,x_test,axis=0)
x=np.concatenate([x_train,x_test],axis=0)
y=np.concatenate([y_train,y_test],axis=0)

x = x.reshape(70000,28*28)
# print(x.shape)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# print(x.shape)

lda = LinearDiscriminantAnalysis(n_components=1)
x= lda.fit_transform(x,y)

print(x.shape,y.shape)
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,random_state=100)

