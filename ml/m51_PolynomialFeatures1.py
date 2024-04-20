import numpy as np
from sklearn.preprocessing import PolynomialFeatures        #다항식

x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]    제곱 -> 0,0,1
#  [2 3]    제곱 -> 4,6(2*3),9
#  [4 5]    제곱 ->16,20(4*5),25
#  [6 7]]   제곱 ->36,42(6*7),49

pf = PolynomialFeatures(degree=2,#제곱
                        include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]
pf = PolynomialFeatures(degree=3,#세제곱
                        #include_bias=False
                        )
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12(2*2*3).  18(2*3*3).  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

print("------------------------------------------------------------------")
x = np.arange(12).reshape(4,3)

pf = PolynomialFeatures(degree=2,#세제곱
                        include_bias=False
                        )
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  9.  10.  11.  81.  90.  99. 100. 110. 121.]]