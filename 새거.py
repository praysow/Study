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
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

MAX_PIXEL_VALUE = 65535

class threadsafe_iter:
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

def FCN(nClasses, input_height=128, input_width=128, n_filters=16, dropout=0.1, batchnorm=True, n_channels=3):
    img_input = Input(shape=(input_height, input_width, n_channels))

    # Block 1
    x = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block1_conv1')(img_input)
    x1 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block1_conv2')(x)
    # drop = Dropout(dropout)(x1)
    x2 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block1_conv3')(x1)
    
    # Block 2
    x3 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv4')(x2)
    x4 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv5')(x3)
    x5 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv6')(x4)
    x6 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv7')(x5)
    batch = BatchNormalization()(x6)
    x7 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv8')(batch)
    x8 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv9')(x7)
    x9 = Conv2D(n_filters, (3, 3), activation='swish', padding='same', name='block2_conv10')(x8)
    # Out
    o = (Conv2D(nClasses, (3, 3), activation='swish', padding='same', name="Out"))(x9)
    model = Model(img_input, o)

    return model

MAX_PIXEL_VALUE = 65535

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)

    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy
save_name = 'base_line'
N_FILTERS = 16
N_CHANNELS = 3
EPOCHS = 100
BATCH_SIZE = 10
IMAGE_SIZE = (256, 256)
MODEL_NAME = 'fcn'
RANDOM_STATE = random.randint(1,300)
INITIAL_EPOCH = 0

IMAGES_PATH = 'c:/_data/aifac/sanbul/train_img/train_img/'
MASKS_PATH = 'c:/_data/aifac/sanbul/train_mask/train_mask/'

OUTPUT_DIR = 'c:/_data/aifac/sanbul/'
WORKERS = 24

EARLY_STOP_PATIENCE = 20

CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

CUDA_DEVICE = 0

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

train_meta = pd.read_csv('c:/_data/aifac/sanbul/train_meta.csv')
test_meta = pd.read_csv('c:/_data/aifac/sanbul/test_meta.csv')

x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

images_train = [os.path.join(IMAGES_PATH, image) for image in x_tr['train_img'] ]
masks_train = [os.path.join(MASKS_PATH, mask) for mask in x_tr['train_mask'] ]

images_validation = [os.path.join(IMAGES_PATH, image) for image in x_val['train_img'] ]
masks_validation = [os.path.join(MASKS_PATH, mask) for mask in x_val['train_mask'] ]

train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")
validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, image_mode="762")

model = FCN(1, input_height=IMAGE_SIZE[0], input_width=IMAGE_SIZE[1], n_filters=N_FILTERS, n_channels=N_CHANNELS)
model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)
rlr = ReduceLROnPlateau(monitor='val_loss',patience=10,mode='auto',verbose=1,factor=0.5)

print('---model 훈련 시작---')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(images_train) // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(images_validation) // BATCH_SIZE,
    callbacks=[checkpoint, es,rlr],
    epochs=EPOCHS,
    workers=WORKERS,
    initial_epoch=INITIAL_EPOCH,
    
)
print('---model 훈련 종료---')

print('가중치 저장')
model_weights_output = os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)
model.save_weights(model_weights_output)
print("저장된 가중치 명: {}".format(model_weights_output))

y_pred_dict = {}

for i in test_meta['test_img']:
    img = get_img_762bands(f'c:/_data/aifac/sanbul/test_img/test_img/{i}')
    y_pred = model.predict(np.array([img]), batch_size=1,verbose=0)

    y_pred = np.where(y_pred[0, :, :, 0] > 0.25, 1, 0)
    y_pred = y_pred.astype(np.uint8)
    y_pred_dict[i] = y_pred

joblib.dump(y_pred_dict, 'c:/_data/aifac/bull11.pkl')

loss = history.history['loss']
accuracy = history.history['accuracy']

print("Training Loss:", loss)
print("Training Accuracy:", accuracy)
print(RANDOM_STATE)
