import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# data
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.9,random_state=47)

sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)

# model 
def self_Voting(models:list, x_train, x_test, y_train, y_test, voting='hard'):
    pred_list = []
    for model in models:
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
        pred_list.append(pred)
        
    # print(pred_list)
    pred_list = np.asarray(pred_list)
    # print(pred_list)
    from scipy.stats import mode
    final_pred = mode(pred_list)[0]
    acc = accuracy_score(final_pred,y_test)
    print("ACC: ",acc)
    # print(final_pred)
    return final_pred
    
model = VotingClassifier([
    ('LR',LogisticRegression()),
    ('RF',RandomForestClassifier()),
    ('XGB',XGBClassifier()),
    ], voting='soft')

y_predict = self_Voting([LogisticRegression(),RandomForestClassifier(),XGBClassifier()],x_train,x_test,y_train,y_test)

# Score:  0.9649122807017544
# ACC:  0.9649122807017544

# VotingClassifier hard
# ACC:  0.9649122807017544

# VotingClassifier soft
# ACC:  0.9649122807017544