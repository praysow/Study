#테스트폴더 쓰진말고 train폴더로
#변환시간도 체크하기

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import time

start_time = time.time()
path = "C:\\_data\\kaggle\\cat-and-dog-classification-harper2022\\"
train_path = path+"train\\"
test_path = path+"test\\"

BATCH_SIZE = int(1000)
IMAGE_SIZE = int(130)

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    # fill_mode='nearest'
)

xy_train_data = train_data_gen.flow_from_directory(
    train_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,        #batch_size 너무 크게주면 에러나옴
    class_mode='binary',
    shuffle=False
)

test_data_gen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_data_gen.flow_from_directory(
    test_path,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=99999,
    class_mode='binary',
    shuffle=False
)

x = []
y = []
failed_i = []

for i in range(int(20000 / BATCH_SIZE)):
    try: 
        xy_data = xy_train_data.next()
        new_x = xy_data[0]
        new_y = xy_data[1]
        if i==0:
            x = np.array(new_x)
            y = np.array(new_y)
            continue
        
        x = np.vstack([x,new_x])
        y = np.hstack([y,new_y])
        print("i: ",i)
        print(f"{x.shape=}\n{y.shape=}")
    except:
        print("failed i: ",i)
        failed_i.append(i)
        
print(failed_i) # [70, 115]

print(f"{x.shape=}\n{y.shape=}")    # x.shape=(1000, 200, 200, 3) y.shape=(1000,)

save_path = path+f"data_{IMAGE_SIZE}px"
np.save(save_path+"_x.npy",arr=x)
np.save(save_path+"_y.npy",arr=y)
np.save(save_path+"_test.npy",arr=test_data[0][0])