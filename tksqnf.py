import os
import warnings
warnings.filterwarnings("ignore")
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import sys
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import threading
import random
import rasterio
import os
import numpy as np
import sys
from sklearn.utils import shuffle as shuffle_lists
from keras.models import *
from keras.layers import *
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
import joblib

"""&nbsp;

## 사용할 함수 정의
"""

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE

    return img

def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg



@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):

    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0
    # 데이터 shuffle
    while True:

        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1


        for img_path, mask_path in zip(images_path, masks_path):

            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []

# MobileNetV3 모델 정의
def MobileNetV3(nClasses, input_height=128, input_width=128, alpha=1.0, dropout=0.1, activation='swish'):
    backbone = keras.applications.MobileNetV3Small(input_shape=(input_height, input_width, 3), alpha=alpha, minimalistic=True, include_top=False, weights=None)

    # Freeze the backbone layers
    backbone.trainable = False

    # Get the output tensor from a layer of the backbone
    x = backbone.output

    # Add more layers as needed for segmentation
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)

    # Final segmentation layer
    outputs = Conv2D(nClasses, (1, 1), activation='softmax')(x)

    # Define the model
    model = Model(inputs=backbone.input, outputs=outputs)

    return model

# 두 샘플 간의 유사성 metric
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

# 픽셀 정확도를 계산 metric
def pixel_accuracy(y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)

    if sum_t == 0:
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy

"""&nbsp;

## parameter 설정
"""

# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('c:/_data/aifac/sanbul/train_meta.csv')
test_meta = pd.read_csv('c:/_data/aifac/sanbul/test_meta.csv')

r = random.randint(1,300)
# 저장 이름
save_name = 'base_line'

N_CLASSES = 2 # 클래스 수 지정
EPOCHS = 1 # 훈련 epoch 지정
BATCH_SIZE = 2 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
RANDOM_STATE = r # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'c:/_data/aifac/sanbul/train_img/train_img/'
MASKS_PATH = 'c:/_data/aifac/sanbul/train_mask/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/_data/aifac/sanbul/'
WORKERS = 24         #코어수

# 조기종료
EARLY_STOP_PATIENCE = 8

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format('MobileNetV3', save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format('MobileNetV3', save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0

# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass

try:
    np.random.bit_generator = np.random._bit_generator
except:
    pass


# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# train : val 지정 및 generator
images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")


# model 불러오기
model = MobileNetV3(N_CLASSES, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1])
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)

"""&nbsp;

## model 훈련
"""

print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH,
    
)
print('---model 훈련 종료---')

"""&nbsp;

## model save
"""

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

"""## inference

- 학습한 모델 불러오기
"""

model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

model.load_weights('c:/_data/aifac/sanbul/model_unet_base_line_final_weights.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장

"""

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'c:/_data/aifac/sanbul/test_img/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1,verbose=0)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'c:/_data/aifac/bull8.pkl')

# 모델 훈련 후에 history 객체에서 손실 및 정확도 값을 가져와서 출력
loss = history.history['loss']
# val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']

# 손실과 정확도 값을 출력
print("Training Loss:", loss)
# print("Validation Loss:", val_loss)
print("Training Accuracy:", accuracy)
# print("Validation Accuracy:", val_accuracy)
print(r)
