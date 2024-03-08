import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso#(L1,L2규제와 비슷한것)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

x,y = load_linnerud(return_X_y=True)
# print(x.shape,y.shape)(20, 3) (20, 3)

# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=8)
#2.모델
model = RandomForestRegressor()
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
# print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
#2.모델
model = Ridge()
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
# print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
#2.모델
model = LinearRegression()
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
# print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
#2.모델
model = XGBRegressor()
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
#2.모델
model = CatBoostRegressor(loss_function='MultieRMSE',verbose=0)   # 에러
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
#2.모델
# model = LGBMRegressor() #에러
# #3.훈련
# model.fit(x,y)
# #4.결과예측
# score = model.score(x,y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어:',
#       round(mean_absolute_error(y,y_pred),4))
# print(model.predict([[2,110,43]]))  #[[152.42  34.04  64.  ]]
# 2.모델
from sklearn.multioutput import MultiOutputRegressor
model =MultiOutputRegressor(CatBoostRegressor(verbose=0)) #에러
#3.훈련
model.fit(x,y)
#4.결과예측
score = model.score(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어:',
      round(mean_absolute_error(y,y_pred),4))
print(model.predict([[2,110,43]]))
'''
RandomForestRegressor 스코어: 3.4878
Ridge 스코어: 7.4569
LinearRegression 스코어: 7.4567
XGBRegressor 스코어: 0.0008
[[138.0005    33.002136  67.99897 ]]
'''