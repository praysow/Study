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
# print(img.shape)   # (1, 150, 150, 3)

#############################여기부터 증폭################################
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)
it = datagen.flow(img, 
                  batch_size=1,
                  )
# fig,ax = plt.subplots(nrows=1,ncols=5, figsize=(10,10)) #subplots은 여러장의 그림을 한번에 부르는것 (1행,5열)

# for i in range(10):
#     batch = it.next()       #it 은 이미지를 반전시켜라 next는 다음도 반전시켜라
#     image = batch[0]
#     ax[i].imshow(img)
#     ax[i].axis('off')
# plt.show()

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(10, 10))      #ax도 2행5열로 바꿔야한다

for i in range(5):
    batch = it.next()
    image = batch[0].astype('uint8')  # imshow 함수는 uint8 형식의 이미지를 기대하므로 형변환 추가
    ax[i].imshow(image)
    ax[i].axis('off')

plt.show()
