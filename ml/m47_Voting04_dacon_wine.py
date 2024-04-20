import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score,log_loss
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
#1. 데이터
path= "c:\_data\dacon\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['quality'], axis=1)
y= train_csv['quality']

y -= 3

lb=LabelEncoder()
lb.fit(x['type'])
x['type'] =lb.transform(x['type'])
test_csv['type'] =lb.transform(test_csv['type'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

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

# Score:  0.5454545454545454
# ACC:  0.5454545454545454

# VotingClassifier hard
# ACC:  0.6763636363636364

# VotingClassifier soft
# ACC:  0.6754545454545454