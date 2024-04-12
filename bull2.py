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
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
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
import time
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
s_t = time.time()
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
    img = rasterio.open(path).read((7,6,5)).transpose((1, 2, 0))
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


"""&nbsp;

## parameter 설정
"""

#############################################모델################################################

#Default Conv2D
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Attention Gate
def attention_gate(F_g, F_l, inter_channel):
    """
    An attention gate.

    Arguments:
    - F_g: Gating signal typically from a coarser scale.
    - F_l: The feature map from the skip connection.
    - inter_channel: The number of channels/filters in the intermediate layer.
    """
    # Intermediate transformation on the gating signal
    W_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_g)
    W_g = BatchNormalization()(W_g)

    # Intermediate transformation on the skip connection feature map
    W_x = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(F_l)
    W_x = BatchNormalization()(W_x)

    # Combine the transformations
    psi = Activation('relu')(add([W_g, W_x]))
    psi = Conv2D(1, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid')(psi)

    # Apply the attention coefficients to the feature map from the skip connection
    return multiply([F_l, psi])

from keras.applications import VGG16
def get_pretrained_attention_unet(input_height=256, input_width=256, nClasses=1, n_filters=16, dropout=0.5, batchnorm=True, n_channels=3):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_height, input_width, n_channels))
    
    # Define the inputs
    inputs = base_model.input
    
    # Use specific layers from the VGG16 model for skip connections
    s1 = base_model.get_layer("block1_conv2").output
    s2 = base_model.get_layer("block2_conv2").output
    s3 = base_model.get_layer("block3_conv3").output
    s4 = base_model.get_layer("block4_conv3").output
    bridge = base_model.get_layer("block5_conv3").output
    
    # Decoder with attention gates
    d1 = UpSampling2D((2, 2))(bridge)
    d1 = concatenate([d1, attention_gate(d1, s4, n_filters*8)])
    d1 = conv2d_block(d1, n_filters*8, kernel_size=3, batchnorm=batchnorm)
    
    d2 = UpSampling2D((2, 2))(d1)
    d2 = concatenate([d2, attention_gate(d2, s3, n_filters*4)])
    d2 = conv2d_block(d2, n_filters*4, kernel_size=3, batchnorm=batchnorm)
    
    d3 = UpSampling2D((2, 2))(d2)
    d3 = concatenate([d3, attention_gate(d3, s2, n_filters*2)])
    d3 = conv2d_block(d3, n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    d4 = UpSampling2D((2, 2))(d3)
    d4 = concatenate([d4, attention_gate(d4, s1, n_filters)])
    d4 = conv2d_block(d4, n_filters, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(nClasses, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_model(model_name, nClasses=1, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    
    if model_name == 'pretrained_attention_unet':
        model = get_pretrained_attention_unet
        
        
    return model(
            nClasses      = nClasses,
            input_height  = input_height,
            input_width   = input_width,
            n_filters     = n_filters,
            dropout       = dropout,
            batchnorm     = batchnorm,
            n_channels    = n_channels
        )

# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('c:/_data/aifac/sanbul/train_meta.csv')
test_meta = pd.read_csv('c:/_data/aifac/sanbul/test_meta.csv')

r = random.randint(1,300)
# 저장 이름
save_name = 'base_line'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 30 # 훈련 epoch 지정
BATCH_SIZE = 6 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pretrained_attention_unet' # 모델 이름
RANDOM_STATE = 6 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'c:/_data/aifac/sanbul/train_img/train_img/'
MASKS_PATH = 'c:/_data/aifac/sanbul/train_mask/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/_data/aifac/sanbul/'
WORKERS = 18         #코어수

# 조기종료
EARLY_STOP_PATIENCE = 5

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_bull21.h5'.format(MODEL_NAME, save_name)

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

#miou metric
def miou(y_true, y_pred, smooth=1e-6):
    THESHOLDS=0.25 # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

# model 불러오기

model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
lr = 0.01
# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

"""&nbsp;

## model 훈련
"""
model.load_weights('c:/_data/aifac/sanbul/attention/bull20.hdf5')
print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es, rlr],
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

# model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
# model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.summary()

# model.load_weights('c:/_data/aifac/sanbul/jj/bull5.h5')

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

joblib.dump(y_pred_dict, 'c:/_data/aifac/bull21.pkl')

# 모델 훈련 후에 history 객체에서 손실 및 정확도 값을 가져와서 출력
loss = history.history['loss']
# val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# 손실과 정확도 값을 출력
print("Training Loss:", loss)
# print("Validation Loss:", val_loss)
print("Training Accuracy:", accuracy)
# print("Validation Accuracy:", val_accuracy)
print(r)
e_t = time.time()
print('time:',e_t-s_t)

'''
Training Loss: [6.810005288571119e-05]
Training Accuracy: [0.9999902844429016]

Training Loss: 6.772846973035485e-05
Training Accuracy:0.999987781047821

Training Loss:  6.597369792871177e-05
Training Accuracy: 0.9999881386756897

Training Loss: 6.484350888058543e-05
Training Accuracy: 0.9999884366989136   30에포 배치6 랜덤 6

18번 50에포에 배치8
Training Loss: [6.961504550417885e-05, 7.061346695991233e-05, 7.075791654642671e-05, 7.030123379081488e-05, 6.995083822403103e-05, 7.049275882309303e-05, 7.025365630397573e-05, 7.004137296462432e-05, 7.015911978669465e-05, 6.971850962145254e-05, 7.022648060228676e-05, 6.590542034246027e-05, 6.555483560077846e-05, 6.573455175384879e-05, 6.554713763762265e-05, 6.55621915939264e-05, 6.543700146721676e-05, 6.545506039401516e-05, 6.484983168775216e-05, 6.576735177077353e-05, 6.518502777907997e-05, 6.405001477105543e-05, 6.432585360016674e-05, 6.402838334906846e-05, 6.391163333319128e-05, 6.382789433700964e-05, 6.405821477528661e-05, 6.395939999492839e-05, 6.402731378329918e-05, 6.385074084391817e-05, 6.416073301807046e-05, 6.352583295665681e-05, 6.364865839714184e-05, 6.332859629765153e-05, 6.352242780849338e-05, 6.349337490973994e-05, 6.353040225803852e-05, 6.341726111713797e-05, 6.347223825287074e-05, 6.340545951388776e-05, 6.354704964905977e-05, 6.330466567305848e-05, 6.316023063845932e-05, 6.366874004015699e-05, 6.324250716716051e-05, 6.364266300806776e-05, 6.348032911773771e-05, 6.321888213278726e-05, 6.324725109152496e-05, 6.337610102491453e-05]
Training Accuracy: [0.9999852180480957, 0.9999852180480957, 0.9999854564666748, 0.9999853372573853, 0.9999851584434509, 0.9999856948852539, 0.9999853372573853, 0.99998539686203, 0.9999851584434509, 0.9999859929084778, 0.9999854564666748, 0.9999861121177673, 0.9999867081642151, 0.9999864101409912, 0.999986469745636, 0.9999868869781494, 0.9999873638153076, 0.9999863505363464, 0.9999868273735046, 0.9999865889549255, 0.9999863505363464, 0.9999878406524658, 0.9999874234199524, 0.9999874234199524, 0.9999873638153076, 0.9999875426292419, 0.9999872446060181, 0.9999875426292419, 0.9999875426292419, 0.9999874830245972, 0.9999873638153076, 0.9999873638153076, 0.9999872446060181, 0.9999879002571106, 0.9999873638153076, 0.9999871253967285, 0.9999871850013733, 0.9999874830245972, 0.9999877214431763, 0.9999878406524658, 0.9999871253967285, 0.9999870657920837, 0.9999873638153076, 0.9999874830245972, 0.9999872446060181, 0.9999874234199524, 0.9999880194664001, 0.999987006187439, 0.9999877214431763, 0.9999871253967285]

19번 30에포 배치6 랜덤 8
loss: 6.3969e-05 - accuracy: 1.0000 - val_loss: 7.3512e-05 - val_accuracy: 1.0000 - lr: 9.0000e-04
20번   30에포 배치6 랜덤 6 0.922
loss: 4.8133e-05 - accuracy: 1.0000 - val_loss: 5.5089e-05 - val_accuracy: 1.0000 - lr: 9.0000e-04
21번

'''