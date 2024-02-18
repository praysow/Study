import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]
                     ])

data = data.transpose()
data.columns = ['x1','x2','x3','x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# data = data.interpolate()

# print(data.isnull())
# print(data.isnull().sum())
# print(data.info())
# print(data.dropna(axis=1))
#2-1.특정값 평균
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 중위값
med = data.median()
print(med)
data3 = data.fillna(med)
print(data3)

#2-3. 특정값 - 0
data4_2 = data.fillna(777)
print(data4_2)

#2-4. 특정값 = ffill, bfill
data5 = data.ffill()
data6 = data.bfill()
print(data5)
print(data6)

#####특정 칼럼만######
means = data['x1'].mean()
print(means)
print(data)

meds = data['x4'].median()
print(meds)

ff= data['x2'].ffill()

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(meds)
data['x2'] = data['x2'].ffill()
