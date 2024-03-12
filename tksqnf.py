import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import rasterio
import numpy as np
import joblib
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

# 데이터 경로 및 설정
train_meta = pd.read_csv('c:/_data/aifac/sanbul/train_meta.csv')
test_meta = pd.read_csv('c:/_data/aifac/sanbul/test_meta.csv')
images_dir = 'c:/_data/aifac/sanbul/train_img/train_img/'
masks_dir = 'c:/_data/aifac/sanbul/train_mask/train_mask/'
test_dir = 'c:/_data/aifac/sanbul/test_img/test_img/'

# 이미지 불러오기 함수 정의
def load_image(image_path):
    with rasterio.open(image_path) as src:
        img = src.read().transpose((1, 2, 0))
        img = np.float32(img) / 65535.0 # 최대 픽셀 값으로 정규화
    return img

# 데이터 불러오기 및 크기 조정 함수 정의
def load_data(images_meta, masks_meta, images_dir, masks_dir, img_size=(256, 256)):
    images = []
    masks = []
    for img_name, mask_name in zip(images_meta['train_img'], masks_meta['train_mask']):
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)
        img = load_image(img_path)
        mask = load_image(mask_path)
        img = resize(img, img_size, mode='constant', preserve_range=True)
        mask = resize(mask, img_size, mode='constant', preserve_range=True)
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

# 학습 데이터 불러오기
X_train, y_train = load_data(train_meta, train_meta, images_dir, masks_dir)

# 테스트 데이터 불러오기
X_test, _ = load_data(test_meta, test_meta, test_dir, None) # 테스트 데이터에는 마스크가 없으므로 두 번째 인자는 None으로 처리

# 이미지 크기 출력
print("학습 데이터 이미지 크기:", X_train.shape)
print("학습 데이터 마스크 크기:", y_train.shape)
print("테스트 데이터 이미지 크기:", X_test.shape)
