import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score,log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

#1.데이터
path= "c:\_data\dacon\dechul\\"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['대출등급', '최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
test= test.drop(['최근_2년간_연체_횟수', '총연체금액', '연체계좌수'],axis=1)
test['대출목적'] = test['대출목적'].replace('결혼', '휴가')
# train.drop(train.index[34488], inplace=True)
y= train['대출등급']

z = test[test['대출목적'].str.contains('결혼')]
lb = LabelEncoder()
columns_to_encode = ['대출기간', '근로기간', '주택소유상태', '대출목적']

for column in columns_to_encode:
    x[column] = lb.fit_transform(x[column])
    test[column] = lb.transform(test[column])

y = lb.fit_transform(train['대출등급'])
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
from sklearn.preprocessing import MinMaxScaler
sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)
from xgboost import XGBClassifier
# model 
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
model = VotingClassifier([
    ('LR',LogisticRegression()),
    ('RF',RandomForestClassifier()),
    ('XGB',XGBClassifier()),
    ], voting='hard')

# fit & pred
model.fit(x_train,y_train,)

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)