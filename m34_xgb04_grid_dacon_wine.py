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
#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']

lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size= 0.9193904973982694, random_state=1909,
                                            stratify=y)
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
n_components=1의 정확도: 0.5045045045045045
[0.31723214]
n_components=2의 정확도: 0.6036036036036037
[0.31723214 0.52790111]
n_components=3의 정확도: 0.6463963963963963
[0.31723214 0.52790111 0.65866185]
n_components=4의 정확도: 0.6463963963963963
[0.31723214 0.52790111 0.65866185 0.73890253]
n_components=5의 정확도: 0.668918918918919
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838 ]
n_components=6의 정확도: 0.6599099099099099
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799]
n_components=7의 정확도: 0.6576576576576577
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621]
n_components=8의 정확도: 0.6801801801801802
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621 0.93727421]
n_components=9의 정확도: 0.6644144144144144
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621 0.93727421 0.96657866]
n_components=10의 정확도: 0.6869369369369369
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621 0.93727421 0.96657866 0.98797477]
n_components=11의 정확도: 0.6891891891891891
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621 0.93727421 0.96657866 0.98797477 0.99788499]
n_components=12의 정확도: 0.6936936936936937
[0.31723214 0.52790111 0.65866185 0.73890253 0.7993838  0.85004799
 0.89542621 0.93727421 0.96657866 0.98797477 0.99788499 1.        ]
'''