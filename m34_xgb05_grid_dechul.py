import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression,SGDClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import LabelEncoder
#1.데이터
path= "c:\_data\dacon\dechul\\"
train_csv=pd.read_csv(path+"train.csv",index_col=0)
test_csv=pd.read_csv(path+"test.csv",index_col=0)
sample_csv=pd.read_csv(path+"sample_submission.csv")
x= train_csv.drop(['대출등급'],axis=1)
y= train_csv['대출등급']


lb=LabelEncoder()
lb.fit(x['대출기간'])
x['대출기간'] = lb.transform(x['대출기간'])
lb.fit(x['근로기간'])
x['근로기간'] = lb.transform(x['근로기간'])
lb.fit(x['주택소유상태'])
x['주택소유상태'] = lb.transform(x['주택소유상태'])
lb.fit(x['대출목적'])
x['대출목적'] = lb.transform(x['대출목적'])

lb.fit(test_csv['대출기간'])
test_csv['대출기간'] =lb.transform(test_csv['대출기간'])

lb.fit(test_csv['근로기간'])
test_csv['근로기간'] =lb.transform(test_csv['근로기간'])

lb.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] =lb.transform(test_csv['주택소유상태'])

lb.fit(test_csv['대출목적'])
test_csv['대출목적'] =lb.transform(test_csv['대출목적'])


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85,random_state=100 ,
                                              #  stratify=y
                                               )

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ra = np.arange(1, min(x_train.shape) + 1)

for n_components in ra:
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    evr = pca.explained_variance_ratio_

    # PCA 모델 학습 및 평가
    model = RandomForestClassifier()
    model.fit(x_train_pca, y_train)
    acc = model.score(x_test_pca, y_test)
    print(f'n_components={n_components}의 정확도:', acc)
    # print(evr)
    # print(sum(evr))
    evr_cumsum = np.cumsum(evr)      #누적합
    print(evr_cumsum)
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
'''
n_components=1의 정확도: 0.2385600553824853
[0.18126384]
n_components=2의 정확도: 0.2817583939079266
[0.18126384 0.28074617]
n_components=3의 정확도: 0.3856697819314642
[0.18126384 0.28074765 0.36886393]
n_components=4의 정확도: 0.4051921079958463
[0.18126384 0.28074765 0.36886393 0.45260852]
n_components=5의 정확도: 0.4189685012114919
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091]
n_components=6의 정확도: 0.43461405330564207
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026]
n_components=7의 정확도: 0.43572170301142266
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543]
n_components=8의 정확도: 0.43433714087919695
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675]
n_components=9의 정확도: 0.4392523364485981
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675 0.82120882]
n_components=10의 정확도: 0.43516787815853236
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675 0.82120882 0.88617456]
n_components=11의 정확도: 0.4402215299411561
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675 0.82120882 0.88617456 0.94381965]
n_components=12의 정확도: 0.44430598823122186
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675 0.82120882 0.88617456 0.94381965 0.97317272]
n_components=13의 정확도: 0.6353063343717549
[0.18126384 0.28074765 0.36886393 0.45260852 0.53042091 0.60783026
 0.68367543 0.75481675 0.82120882 0.88617456 0.94381965 0.97317272
 1.        ]
'''