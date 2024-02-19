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
#1. 데이터

path= "c:\_data\dacon\ddarung\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

train_csv=train_csv.fillna(train_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())                         #test는 dropna를 하면 안되고 결측치를 변경해줘야한다
# test_csv=test_csv.fillna(0)

x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

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
    model = RandomForestRegressor()
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
n_components=1의 정확도: 0.2697235625113683
[0.33588699]
n_components=2의 정확도: 0.4605392188020373
[0.33588699 0.54678185]
n_components=3의 정확도: 0.5233020005457962
[0.33588699 0.54678185 0.68139918]
n_components=4의 정확도: 0.5774414516677755
[0.33588699 0.54678185 0.68139918 0.76022591]
n_components=5의 정확도: 0.6582220824938497
[0.33588699 0.54678185 0.68139918 0.76022591 0.83158143]
n_components=6의 정확도: 0.6674968946049984
[0.33588699 0.54678185 0.68139918 0.76022591 0.83158143 0.89451238]
n_components=7의 정확도: 0.6837674433233867
[0.33588699 0.54678185 0.68139918 0.76022591 0.83158143 0.89451238
 0.94139662]
n_components=8의 정확도: 0.6886748798382076
[0.33588699 0.54678185 0.68139918 0.76022591 0.83158143 0.89451238
 0.94139662 0.98226234]
n_components=9의 정확도: 0.682578653927959
[0.33588699 0.54678185 0.68139918 0.76022591 0.83158143 0.89451238
 0.94139662 0.98226234 1.        ]
'''
