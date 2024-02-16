import numpy as np
aaa= np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
              [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T #(13,2)

def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])
    print("1사분위 :", quartile_1)
    print("q2",q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr:",iqr)
    lower_bound = quartile_1 - (iqr*1.5)    # *1.5 이걸만든 프로그래머가 정한수치이고 직접 조정해도 된다(범위를 조금 늘리기 위해서 해주는것)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) |    # |는 또는이라는 뜻이다
                    (data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 :",outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()


