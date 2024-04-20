import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import accuracy_score,log_loss
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import LabelEncoder
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

from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from xgboost import XGBClassifier
pf = PolynomialFeatures(degree=2,include_bias=False)
x_poly = pf.fit_transform(x)
print(x_poly)

#2.모델
model = XGBClassifier()
model2= XGBClassifier()
#3.훈련
# print('s',x.shape)
# print('s',y.shape)

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
'''
R2 Score: 0.6836363636363636
'''