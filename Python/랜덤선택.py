import numpy as np
from keras.models import Sequential
from keras.layers import Dense


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])


# import random

# random_element = random.choice(x)
# randomdd = random.choice(y)
# print("랜덤으로 선택된 원소:", random_element)

import random
# x=[1,2,3,4,5,6,7,8,9,10]

# 리스트에서 중복 없이 랜덤으로 세 개의 원소 추출
random_elements = random.sample(x,range(7))
randomdd= random.sample(y)

print("랜덤으로 선택된 원소들:", randomdd)
print("랜덤으로 선택된 원소들:", random_elements)

