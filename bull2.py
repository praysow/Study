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

seed_everything(65535)
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
BATCH_SIZE =15 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'pretrained_attention_unet' # 모델 이름
RANDOM_STATE = 8 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = 'c:/_data/aifac/sanbul/train_img/'
MASKS_PATH = 'c:/_data/aifac/sanbul/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = 'c:/_data/aifac/sanbul/'
WORKERS = 12         #코어수

# 조기종료
EARLY_STOP_PATIENCE = 5

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 1
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_bull25.h5'.format(MODEL_NAME, save_name)

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
    THESHOLDS=0.5 # 임계치 기준으로 이진화
    y_pred = tf.cast(y_pred > THESHOLDS, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    
    # mIoU 계산
    iou = (intersection + smooth) / (union + smooth)
    miou = tf.reduce_mean(iou)
    return miou

# model 불러오기

model = get_model(MODEL_NAME, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
lr = 0.001
model.compile(optimizer = Adam(learning_rate=lr), loss = 'binary_crossentropy', metrics = ['accuracy',miou])
model.summary()

# checkpoint 및 조기종료 설정
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
save_best_only=False, mode='auto', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',patience=1,mode='auto',verbose=1,factor=0.5)

"""&nbsp;

## model 훈련
"""
model.load_weights('c:/_data/aifac/sanbul/attention/bull22.hdf5')
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
# lr = 0.01
# model.compile(optimizer = Adam(learning_rate=lr), loss = 'binary_crossentropy', metrics = ['accuracy',miou])
# # model.summary()

# model.load_weights('c:/_data/aifac/sanbul/bull22.h5')

"""## 제출 Predict
- numpy astype uint8로 지정
- 반드시 pkl로 저장

"""

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'c:/_data/aifac/sanbul/test_img/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1,verbose=0)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.5, 1, 0) # 임계값 처리
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'c:/_data/aifac/bull25.pkl')

loss = model.evaluate_generator(validation_generator, steps=len(images_validation) // BATCH_SIZE)
print("Validation Loss:", loss)

# 모델 훈련 후에 history 객체에서 손실 및 정확도 값을 가져와서 출력
loss = history.history['loss']

accuracy = history.history['accuracy']


# 손실과 정확도 값을 출력
print("Training Loss:", loss)
print("Training Accuracy:", accuracy)
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
23번
Validation Loss: [4.93808402097784e-05, 0.9999816417694092, 0.9324949383735657]



























Epoch 1/30
2024-03-23 14:40:46.203191: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
1918/1918 [==============================] - ETA: 0s - loss: 5.0863e-05 - miou: 0.9260  
Epoch 1: loss improved from inf to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_01.hdf5
1918/1918 [==============================] - 662s 342ms/step - loss: 5.0863e-05 - miou: 0.9260 - val_loss: 5.9499e-05 - val_miou: 0.9144 - lr: 0.0100
Epoch 2/30
1918/1918 [==============================] - ETA: 0s - loss: 5.1013e-05 - miou: 0.9244  
Epoch 2: loss did not improve from 0.00005

Epoch 2: ReduceLROnPlateau reducing learning rate to 0.004999999888241291.
1918/1918 [==============================] - 663s 346ms/step - loss: 5.1013e-05 - miou: 0.9244 - val_loss: 5.8382e-05 - val_miou: 0.9258 - lr: 0.0100
Epoch 3/30
1918/1918 [==============================] - ETA: 0s - loss: 4.9047e-05 - miou: 0.9288  
Epoch 3: loss improved from 0.00005 to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_03.hdf5

Epoch 3: ReduceLROnPlateau reducing learning rate to 0.0024999999441206455.
1918/1918 [==============================] - 5246s 3s/step - loss: 4.9047e-05 - miou: 0.9288 - val_loss: 9.3990e-05 - val_miou: 0.8667 - lr: 0.0050
Epoch 4/30
1918/1918 [==============================] - ETA: 0s - loss: 4.7693e-05 - miou: 0.9308  
Epoch 4: loss improved from 0.00005 to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_04.hdf5

Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
1918/1918 [==============================] - 622s 324ms/step - loss: 4.7693e-05 - miou: 0.9308 - val_loss: 5.4212e-05 - val_miou: 0.9265 - lr: 0.0025
Epoch 5/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6870e-05 - miou: 0.9322  
Epoch 5: loss improved from 0.00005 to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_05.hdf5

Epoch 5: ReduceLROnPlateau reducing learning rate to 0.0006249999860301614.
1918/1918 [==============================] - 632s 330ms/step - loss: 4.6870e-05 - miou: 0.9322 - val_loss: 5.2332e-05 - val_miou: 0.9312 - lr: 0.0012
Epoch 6/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6496e-05 - miou: 0.9322
Epoch 6: loss improved from 0.00005 to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_06.hdf5

Epoch 6: ReduceLROnPlateau reducing learning rate to 0.0003124999930150807.
1918/1918 [==============================] - 1492s 778ms/step - loss: 4.6496e-05 - miou: 0.9322 - val_loss: 5.3034e-05 - val_miou: 0.9298 - lr: 6.2500e-04
Epoch 7/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6226e-05 - miou: 0.9326
Epoch 7: loss improved from 0.00005 to 0.00005, saving model to c:/_data/aifac/sanbul\checkpoint-pretrained_attention_unet-base_line-epoch_07.hdf5

Epoch 7: ReduceLROnPlateau reducing learning rate to 0.00015624999650754035.
1918/1918 [==============================] - 657s 342ms/step - loss: 4.6226e-05 - miou: 0.9326 - val_loss: 5.3230e-05 - val_miou: 0.9286 - lr: 3.1250e-04
1918/1918 [==============================] - ETA: 0s - loss: 4.6249e-05 - miou: 0.9330
Epoch 8: loss did not improve from 0.00005

Epoch 8: ReduceLROnPlateau reducing learning rate to 7.812499825377017e-05.        
1918/1918 [==============================] - 585s 305ms/step - loss: 4.6249e-05 - miou: 0.9330 - val_loss: 5.1389e-05 - val_miou: 0.9318 - lr: 1.5625e-04
Epoch 9/30
ul\checkpoint-pretrained_attention_unet-base_line-epoch_09.hdf5

Epoch 9: ReduceLROnPlateau reducing learning rate to 3.9062499126885086e-05.
1918/1918 [==============================] - 601s 313ms/step - loss: 4.6008e-05 - miou: 0.9330 - val_loss: 5.2211e-05 - val_miou: 0.9316 - lr: 7.8125e-05
Epoch 10/30
 103/1918 [>.............................] - ETA: 8:38 - loss: 3.7484e-05 - miou: 0 104/1918 [>.............................] - ETA: 8:36 - loss: 3.7405e-05 - miou: 0 105/1918 [>.............................] - ETA: 8:35 - loss: 3.7345e-05 - miou: 0 106/1918 [>.............................] - ETA: 8:36 - loss: 3.7385e-05 - miou: 0 107/1918 [>.............................] - ETA: 8:36 - loss: 3.7271e-05 - miou: 0 108/1918 [>.............................] - ETA: 8:36 - loss: 3.7707e-05 - miou: 0 109/1918 [>.............................] - ETA: 8:36 - loss: 3.7637e-05 - miou: 0 110/1918 [>.............................] - ETA: 8:35 - loss: 3.7554e-05 - miou: 0 111/1918 [>.............................] - ETA: 8:35 - loss: 3.7525e-05 - miou: 0 112/1918 [>.............................] - ETA: 8:35 - loss: 3.7539e-05 - miou: 0 113/1918  1201918/1918 [==============================] - ETA: 0s - loss: 4.6245e-05 - miou: 0.9330  ......................] - ETA: 8:35 - loss: 3.7412e-05 - miou: 0.9349
Epoch 10: loss did not improve from 0.00005

Epoch 10: ReduceLROnPlateau reducing learning rate to 1.9531249563442543e-05.
1918/1918 [==============================] - 728s 380ms/step - loss: 4.6245e-05 - miou: 0.9330 - val_loss: 5.3577e-05 - val_miou: 0.9310 - lr: 3.9062e-05
Epoch 11/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6061e-05 - miou: 0.9331  
Epoch 11: loss did not improve from 0.00005

Epoch 11: ReduceLROnPlateau reducing learning rate to 9.765624781721272e-06.
1918/1918 [==============================] - 735s 383ms/step - loss: 4.6061e-05 - miou: 0.9331 - val_loss: 5.3625e-05 - val_miou: 0.9300 - lr: 1.9531e-05
Epoch 12/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6292e-05 - miou: 0.9327  
Epoch 12: loss did not improve from 0.00005

Epoch 12: ReduceLROnPlateau reducing learning rate to 4.882812390860636e-06.
1918/1918 [==============================] - 734s 383ms/step - loss: 4.6292e-05 - miou: 0.9327 - val_loss: 5.1254e-05 - val_miou: 0.9318 - lr: 9.7656e-06
Epoch 13/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6046e-05 - miou: 0.9330  
Epoch 13: loss did not improve from 0.00005

Epoch 13: ReduceLROnPlateau reducing learning rate to 2.441406195430318e-06.
1918/1918 [==============================] - 753s 393ms/step - loss: 4.6046e-05 - miou: 0.9330 - val_loss: 6.0403e-05 - val_miou: 0.9306 - lr: 4.8828e-06
Epoch 14/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6239e-05 - miou: 0.9334
Epoch 14: loss did not improve from 0.00005

Epoch 14: ReduceLROnPlateau reducing learning rate to 1.220703097715159e-06.
1918/1918 [==============================] - 759s 396ms/step - loss: 4.6239e-05 - miou: 0.9334 - val_loss: 4.5632e-05 - val_miou: 0.9301 - lr: 2.4414e-06
Epoch 15/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6106e-05 - miou: 0.9331
Epoch 15: loss did not improve from 0.00005

Epoch 15: ReduceLROnPlateau reducing learning rate to 6.103515488575795e-07.
1918/1918 [==============================] - 763s 398ms/step - loss: 4.6106e-05 - miou: 0.9331 - val_loss: 5.6218e-05 - val_miou: 0.9291 - lr: 1.2207e-06
Epoch 16/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6228e-05 - miou: 0.9325
Epoch 16: loss did not improve from 0.00005

Epoch 16: ReduceLROnPlateau reducing learning rate to 3.0517577442878974e-07.
1918/1918 [==============================] - 741s 386ms/step - loss: 4.6228e-05 - miou: 0.9325 - val_loss: 5.8644e-05 - val_miou: 0.9330 - lr: 6.1035e-07
Epoch 17/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6157e-05 - miou: 0.9330  
Epoch 17: loss did not improve from 0.00005

1918/1918 [==============================] - 632s 330ms/step - loss: 4.6157e-05 - miou: 0.9330 - val_loss: 4.6871e-05 - val_miou: 0.9300 - lr: 3.0518e-07
Epoch 18/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6174e-05 - miou: 0.9330
Epoch 18: loss did not improve from 0.00005

Epoch 18: ReduceLROnPlateau reducing learning rate to 7.629394360719743e-08.       
1918/1918 [==============================] - 633s 330ms/step - loss: 4.6174e-05 - miou: 0.9330 - val_loss: 5.1949e-05 - val_miou: 0.9298 - lr: 1.5259e-07
Epoch 19/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6147e-05 - miou: 0.9330
Epoch 19: loss did not improve from 0.00005

Epoch 19: ReduceLROnPlateau reducing learning rate to 3.814697180359872e-08.       
1918/1918 [==============================] - 633s 330ms/step - loss: 4.6147e-05 - miou: 0.9330 - val_loss: 5.2839e-05 - val_miou: 0.9315 - lr: 7.6294e-08
Epoch 20/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6254e-05 - miou: 0.9328
Epoch 20: loss did not improve from 0.00005

Epoch 20: ReduceLROnPlateau reducing learning rate to 1.907348590179936e-08.       
1918/1918 [==============================] - 642s 335ms/step - loss: 4.6254e-05 - miou: 0.9328 - val_loss: 5.4389e-05 - val_miou: 0.9310 - lr: 3.8147e-08
Epoch 21/30

ETA: 8:20 - loss: 4.4725e-05 - miou: 0 270/1918 [===>..........................] - ETA: 8:19 - loss: 4.5985e-05 - miou: 0 271/1918 [===>..........................] - ETA: 8:20 - loss: 4.5868e-05 - miou: 0 272/1918 [===>..........................] - ETA: 8:19 - loss: 4.5880e-05 - miou: 0 273/1918 [===>..........................] - ETA: 8:19 - loss: 4.5784e-05 - miou: 0 274/1918 [===>..........................] - ETA: 8:18 - loss: 4.5691e-05 - miou: 0 275/1918 [===>..........................] - ETA: 8:18 - loss: 4.5599e-05 - miou: 0 276/1918 [===>..........................] - ETA: 8:17 - loss: 4.5567e-05 - miou: 0 277/1918 [===>..........................] - ETA: 8:17 - loss: 4.6163e-05 - miou: 0 278/1918 [===>..........................] - ETA: 8:17 - loss: 4.6103e-05 - miou: 0 279/1918 [===>..........................] - ETA: 8:16 - loss: 4.6196e-05 - miou: 0 280/1918 [===>..........................] - ETA: 8:16 - loss: 4.6201e-05 - miou: 0 281/1918 [===>..........................] - ETA: 8:16 - loss: 4.6162e-05 - miou: 0 282/1918 [===>..........................] - ETA: 8:16 - loss: 4.6138e-05 - miou: 0 283/1918 [===>..........................] - ETA: 8:15 - loss: 4.6078e-05 - miou: 0 284/1918 [===>..........................] - ETA: 8:15 - loss: 4.61918/1918 [==============================] - ETA: 0s - loss: 4.6118e-05 - miou: 0.9330  9 - loss: 4.5777e-05 - miou: 0.9345..................] - ETA: 8:14 - loss: 4.6379e-05 - miou: 0.9349
Epoch 21: loss did not improve from 0.00005

Epoch 21: ReduceLROnPlateau reducing learning rate to 9.53674295089968e-09.
1918/1918 [==============================] - 687s 358ms/step - loss: 4.6118e-05 - miou: 0.9330 - val_loss: 5.9938e-05 - val_miou: 0.9313 - lr: 1.9073e-08
Epoch 22/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6104e-05 - miou: 0.9330  
Epoch 22: loss did not improve from 0.00005

Epoch 22: ReduceLROnPlateau reducing learning rate to 4.76837147544984e-09.
1918/1918 [==============================] - 659s 344ms/step - loss: 4.6104e-05 - miou: 0.9330 - val_loss: 5.2682e-05 - val_miou: 0.9307 - lr: 9.5367e-09
Epoch 23/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6286e-05 - miou: 0.9330
Epoch 23: loss did not improve from 0.00005

Epoch 23: ReduceLROnPlateau reducing learning rate to 2.38418573772492e-09.
1918/1918 [==============================] - 743s 387ms/step - loss: 4.6286e-05 - miou: 0.9330 - val_loss: 5.2778e-05 - val_miou: 0.9303 - lr: 4.7684e-09
Epoch 24/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6043e-05 - miou: 0.9332  
Epoch 24: loss did not improve from 0.00005

Epoch 24: ReduceLROnPlateau reducing learning rate to 1.19209286886246e-09.
1918/1918 [==============================] - 690s 360ms/step - loss: 4.6043e-05 - miou: 0.9332 - val_loss: 4.4219e-05 - val_miou: 0.9310 - lr: 2.3842e-09
Epoch 25/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6200e-05 - miou: 0.9329  
Epoch 25: loss did not improve from 0.00005

Epoch 25: ReduceLROnPlateau reducing learning rate to 5.9604643443123e-10.
1918/1918 [==============================] - 712s 371ms/step - loss: 4.6200e-05 - miou: 0.9329 - val_loss: 5.2993e-05 - val_miou: 0.9334 - lr: 1.1921e-09
Epoch 26/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6043e-05 - miou: 0.9333  
Epoch 26: loss did not improve from 0.00005

Epoch 26: ReduceLROnPlateau reducing learning rate to 2.98023217215615e-10.
1918/1918 [==============================] - 1171s 611ms/step - loss: 4.6043e-05 - miou: 0.9333 - val_loss: 6.2367e-05 - val_miou: 0.9289 - lr: 5.9605e-10
Epoch 27/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6052e-05 - miou: 0.9335
Epoch 27: loss did not improve from 0.00005

Epoch 27: ReduceLROnPlateau reducing learning rate to 1.490116086078075e-10.
1918/1918 [==============================] - 731s 381ms/step - loss: 4.6052e-05 - miou: 0.9335 - val_loss: 4.2595e-05 - val_miou: 0.9316 - lr: 2.9802e-10
Epoch 28/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6340e-05 - miou: 0.9328
Epoch 28: loss did not improve from 0.00005

Epoch 28: ReduceLROnPlateau reducing learning rate to 7.450580430390374e-11.
1918/1918 [==============================] - 863s 450ms/step - loss: 4.6340e-05 - miou: 0.9328 - val_loss: 5.5047e-05 - val_miou: 0.9308 - lr: 1.4901e-10
Epoch 29/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6067e-05 - miou: 0.9333  
Epoch 29: loss did not improve from 0.00005

Epoch 29: ReduceLROnPlateau reducing learning rate to 3.725290215195187e-11.
1918/1918 [==============================] - 705s 368ms/step - loss: 4.6067e-05 - miou: 0.9333 - val_loss: 5.4124e-05 - val_miou: 0.9322 - lr: 7.4506e-11
Epoch 30/30
1918/1918 [==============================] - ETA: 0s - loss: 4.6379e-05 - miou: 0.9330  
Epoch 30: loss did not improve from 0.00005

Epoch 30: ReduceLROnPlateau reducing learning rate to 1.8626451075975936e-11.
1918/1918 [==============================] - 738s 385ms/step - loss: 4.6379e-05 - miou: 0.9330 - val_loss: 6.0695e-05 - val_miou: 0.9290 - lr: 3.7253e-11
---model 훈련 종료---
가중치 저장
저장된 가중치 명: c:/_data/aifac/sanbul/model_pretrained_attention_unet_base_line_bull22.h5
Traceback (most recent call last):
File "c:\study\bull.py", line 395, in <module>
'''