import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]    제곱 -> 0,0,1
#  [2 3]    제곱 -> 4,6(2*3),9
#  [4 5]    제곱 ->16,20(4*5),25
#  [6 7]]   제곱 ->36,42(6*7),49

pf = PolynomialFeatures(degree=2,include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)