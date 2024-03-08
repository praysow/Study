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
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from xgboost import XGBClassifier
pf = PolynomialFeatures(degree=2,include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2.모델
model = XGBClassifier()
model2= XGBClassifier()
#3.훈련
print('s',x.shape)
print('s',y.shape)

model.fit(x,y)
model2.fit(x_poly,y)
#4.시각화
# plt.scatter(x,y,color = 'blue',label = 'Original')
# plt.xlabel('x')
# plt.xlabel('y')
# plt.title('Polynomial Regression Example')

# x_plot = np.linspace(-1,1,100).reshape(-1,1)
# x_plot_poly = pf.transform(x_plot)
# y_plot = model.predict(x_plot)
# y_plot2 = model2.predict(x_plot_poly)
# plt.plot(x_plot,y_plot,color = 'red',label = 'Polynomial Regression')
# plt.plot(x_plot,y_plot2,color = 'blue',label = '기냥')
# plt.legend()
# plt.show()

x_train,x_test,y_train,y_test = train_test_split(x_poly,y,train_size=0.9,random_state=1)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = accuracy_score(y_test, y_pred)
print("R2 Score:", r2)