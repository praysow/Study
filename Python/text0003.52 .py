import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array

path = "c:/_data/image/cat_and_dog/train/Cat/1.jpg"
img = load_img(path, target_size=(150, 150))
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0)

#############################여기부터 증폭################################
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
)
it = datagen.flow(img, batch_size=1)

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 10))

for i in range(10):
    batch = it.next()
    image = batch[0].astype('uint8')
    ax[int(i/5)][i%5].imshow(image)  # Fix the indexing for the subplot
    ax[int(i/5)][i%5].axis('off')
# for i in range(5):
#     batch = it.next()
#     image = batch[0].astype('uint8')
#     ax[1, i].imshow(image)  # Fix the indexing for the subplot
#     ax[1, i].axis('off')
# Original image
# ax[1, 0].imshow(arr.astype('uint8'))  # Convert the original image array to uint8 for display
# ax[1, 0].axis('off')

# Display the original image in the second row
plt.show()
