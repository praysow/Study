import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.metrics import accuracy_score, f1_score ,log_loss
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold,GridSearchCV,RandomizedSearchCV,HalvingRandomSearchCV,HalvingGridSearchCV

path= "c:/_data/kaggle/비만/"
train=pd.read_csv(path+"train.csv",index_col=0)
test=pd.read_csv(path+"test.csv",index_col=0)
sample=pd.read_csv(path+"sample_submission.csv")
x= train.drop(['NObeyesdad'],axis=1)
y= train['NObeyesdad']
# print(train.shape,test.shape)   #(20758, 17) (13840, 16)    NObeyesdad
# print(x.shape,y.shape)  #(20758, 16) (20758,)

lb = LabelEncoder()
y = lb.fit_transform(train['NObeyesdad'])
# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])

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