import numpy as np
aaa= np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
              [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T #(13,2)

aaa = aaa.reshape(-1,1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)           #contamination 은 전체 데이터에서 10%이다

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)

# [ 1  1  1  1  1  1  1  1  1  1  1  1  1 -1  1  1  1 -1  1 -1  1  1  1  1  1  1]
#데이터가 한줄로 나오는 이유는 하나의 데이터로 나오기때문이다
#그렇기때문에 컬럼별로 해주는게 더 좋다