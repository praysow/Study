import sys
import tensorflow as tf
print("텐서",tf.__version__,)
print("파이썬",sys.version)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img #이미지를 가져옴
from tensorflow.keras.preprocessing.image import img_to_array #이미지를 수치화

path = "c:/_data/image/cat_and_dog/train/Cat/1.jpg"
img= load_img(path,
                target_size=(150,150)
                )
# print(type(img))
# print(img)
# # plt.imshow(img)
# # plt.show()
arr = img_to_array(img)
# print(arr)
# print(arr.shape)    #(281,300,3)  ->  (150, 150, 3)
# print(type(arr))    #<class 'numpy.ndarray'>

#차원증가
img = np.expand_dims(arr, axis=0)
print(img.shape)   # (1, 150, 150, 3)


