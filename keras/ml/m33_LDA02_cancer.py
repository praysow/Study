#y값이 필요없기때문에 대표적인 비지도 학습중하나이다
#데이터에 0이 많은 것들은 성능이 더 좋게 나올수도 있다.
#보통 PCA를 하기전에 스탠다드스케일러를 사용하면 좋다
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
x= lda.fit_transform(x,y)
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=3,shuffle=True,stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ra = np.arange(1, min(x_train.shape) + 1)
# ra = np.arange(1, 20)

for n_components in ra:
    # pca = PCA(n_components=n_components)
    # x_train_pca = pca.fit_transform(x_train)
    # x_test_pca = pca.transform(x_test)
    evr = lda.explained_variance_ratio_

    # PCA 모델 학습 및 평가
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f'n_components={n_components}의 정확도:', acc)
    print(evr)
    print(sum(evr))
    evr_cumsum = np.cumsum(evr)      #누적합
    print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
# plt.show()
'''
n_components=1의 정확도: 0.9824561403508771
[1.]
1.0
[1.]
'''
