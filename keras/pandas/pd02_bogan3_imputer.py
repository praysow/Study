import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
# print(data)

# #      x1   x2    x3   x4
# # 0   2.0  2.0   2.0  NaN
# # 1   NaN  4.0   4.0  4.0
# # 2   6.0  NaN   6.0  NaN
# # 3   8.0  8.0   8.0  8.0
# # 4  10.0  NaN  10.0  NaN

# from sklearn.impute import SimpleImputer, KNNImputer            #KNN은 가까이 있는 수치를 따라간다,범위에 따라 계속 바뀔수도 있다
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imputer = SimpleImputer()           #디폴트는 평균값
# data = imputer.fit_transform(data) #평균
# print(data)
# imputer = SimpleImputer(strategy='mean')           #디폴트는 평균값
# data2 = imputer.fit_transform(data) #평균
# print(data2)

# imputer = SimpleImputer(strategy='median')
# data4 = imputer.fit_transform(data) #중위
# print(data4)
# imputer = SimpleImputer(strategy='most_frequent')
# data5 = imputer.fit_transform(data) #가장자주나오는놈
# print(data5)
# imputer = SimpleImputer(strategy='constant',fill_value=777)
# data6 = imputer.fit_transform(data) #상수 : 0
# print(data6)

# imputer = KNNImputer()
# data7 = imputer.fit_transform(data)
# print(data7)
# print("+++++++++++++++++++++++++++++++++")
# imputers = IterativeImputer()       #선형회귀 알고리즘
# data8 = imputers.fit_transform(data)
# print(data8)
# 1.22.4
#pip install impyute
from impyute.imputation.cs import mice

# np.float 대신 float 사용
aaa = mice(data.values,
           n = 10,
           seed = 777,
           )
print(aaa)