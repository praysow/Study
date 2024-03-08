import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


plt.rcParams['font.family']='Malgun Gothic'

#1.데이터
np.random.seed(777)
x = 2*np.random.rand(100,1)-1   #1부터 -1까지 난수 생성
y = 3*x**2+2*x+1+np.random.randn(100,1) #0~1사이의 값을 추가해서 노이즈를 추가하는것    y=3x^2+2x+1+노이즈
# print(y)

pf = PolynomialFeatures(degree=2,include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2.모델
model = LinearRegression()
# model = RandomForestRegressor()
model2= RandomForestRegressor()
# model = XGBRegressor()
# model2= XGBRegressor()
#3.훈련
model.fit(x,y)
model2.fit(x_poly,y)
#4.시각화
plt.scatter(x,y,color = 'blue',label = 'Original')
plt.xlabel('x')
plt.xlabel('y')
plt.title('Polynomial Regression Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot,y_plot,color = 'red',label = 'Polynomial Regression')
plt.plot(x_plot,y_plot2,color = 'blue',label = '기냥')
plt.legend()
plt.show()