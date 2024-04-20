import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing

# 데이터 로드
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# # 타겟 변수를 이진 분류로 변환
# median_y = np.median(y)
# y_binary = np.where(y > median_y, 1, 0)

from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from xgboost import XGBRegressor
pf = PolynomialFeatures(degree=2,include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2.모델
model = XGBRegressor()
model2= XGBRegressor()
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

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
'''
R2 Score: 0.8324444517699755
'''