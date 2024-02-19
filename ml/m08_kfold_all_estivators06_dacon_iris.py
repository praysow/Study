from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
#1.데이터
path= "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sample_Submission.csv")
x= train_csv.drop(['species'], axis=1)
y= train_csv['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.86,
                                                    random_state=100,        #346
                                                    #stratify=y_ohe           
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
asd
"""
AdaBoostClassifier 의 정답률: 0.8235294117647058
BaggingClassifier 의 정답률: 0.8235294117647058
BernoulliNB 의 정답률: 0.23529411764705882
CalibratedClassifierCV 의 정답률: 0.7058823529411765
CategoricalNB 의 정답률: 0.8823529411764706
ClassifierChain 실패
ComplementNB 의 정답률: 0.7647058823529411
DecisionTreeClassifier 의 정답률: 0.8235294117647058
DummyClassifier 의 정답률: 0.23529411764705882
ExtraTreeClassifier 의 정답률: 0.8235294117647058
ExtraTreesClassifier 의 정답률: 0.8235294117647058
GaussianNB 의 정답률: 0.8235294117647058
GaussianProcessClassifier 의 정답률: 0.8823529411764706
GradientBoostingClassifier 의 정답률: 0.8235294117647058
HistGradientBoostingClassifier 의 정답률: 0.8235294117647058
KNeighborsClassifier 의 정답률: 0.8823529411764706
LabelPropagation 의 정답률: 0.8823529411764706
LabelSpreading 의 정답률: 0.8823529411764706
LinearDiscriminantAnalysis 의 정답률: 1.0
LinearSVC 의 정답률: 0.8823529411764706
LogisticRegression 의 정답률: 0.8235294117647058
LogisticRegressionCV 의 정답률: 0.8823529411764706
MLPClassifier 의 정답률: 0.9411764705882353
MultiOutputClassifier 실패
MultinomialNB 의 정답률: 0.47058823529411764
NearestCentroid 의 정답률: 0.8235294117647058
NuSVC 의 정답률: 0.8823529411764706
OneVsOneClassifier 실패
OneVsRestClassifier 실패
OutputCodeClassifier 실패
PassiveAggressiveClassifier 의 정답률: 0.47058823529411764
Perceptron 의 정답률: 0.7647058823529411
QuadraticDiscriminantAnalysis 의 정답률: 1.0
RadiusNeighborsClassifier 의 정답률: 0.8823529411764706
RandomForestClassifier 의 정답률: 0.8235294117647058
RidgeClassifier 의 정답률: 0.7058823529411765
RidgeClassifierCV 의 정답률: 0.7058823529411765
SGDClassifier 의 정답률: 0.7058823529411765
SVC 의 정답률: 0.8823529411764706
StackingClassifier 실패
VotingClassifier 실패
"""