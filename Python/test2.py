#R2 0.62이상
from sklearn.datasets import load_diabetes

#1.데이터
datasets= load_diabetes()
x= datasets.data
y= datasets.target

print(x.shape) #(442,18)
print(y.shape) #(442,)

