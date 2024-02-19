import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import lightgbm as lgb
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
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

from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')


n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=123)
#2.모델훈련
for name, algorithm in allAlgorithms:
    try:
        #2.모델
        model = algorithm()
        #3.훈련
       
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        print("ACC:",scores,"\n평균:",round(np.mean(scores),4))

        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_pred)
        print("acc", acc)
    except:
        print(name, '실패')
        continue

'''
AdaBoostClassifier 의 정답률: 0.5075112495673243
BaggingClassifier 의 정답률: 0.8676358601592247
BernoulliNB 의 정답률: 0.3621322256836276
CalibratedClassifierCV 의 정답률: 0.3685012114918657
CategoricalNB 실패
ClassifierChain 실패
ComplementNB 의 정답률: 0.20989961924541364
DecisionTreeClassifier 의 정답률: 0.8353063343717549
DummyClassifier 의 정답률: 0.30834198684665975
ExtraTreeClassifier 의 정답률: 0.4559363101419176
'''