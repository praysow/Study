import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

# 하이퍼파라미터 설정
IMG_SIZE = 224
SEED = 6

# 시드 설정
np.random.seed(SEED)

# 데이터 경로
path = 'c:/_data/dacon/bird'

# 데이터 불러오기
df = pd.read_csv(os.path.join(path, 'train.csv'))

# 데이터 전처리
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])
train, val = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)

# 이미지 데이터 로드 및 전처리
def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None and image.size != 0:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
    return np.array(images)


X_train = load_images(train['img_path'])
X_val = load_images(val['img_path'])
y_train = train['label']
y_val = val['label']

# LightGBM 모델 정의 및 학습
model = LGBMClassifier()
model.fit(X_train.reshape(-1, IMG_SIZE * IMG_SIZE * 3), y_train)

# 검증 데이터에 대한 예측 및 평가
val_preds = model.predict(X_val.reshape(-1, IMG_SIZE * IMG_SIZE * 3))
val_f1_score = f1_score(y_val, val_preds, average='macro')
print("Validation F1 Score:", val_f1_score)

# 테스트 데이터 로드 및 예측
test_df = pd.read_csv(os.path.join(path, 'test.csv'))
X_test = load_images(test_df['img_path'])
test_preds = model.predict(X_test.reshape(-1, IMG_SIZE * IMG_SIZE * 3))

# 결과 저장
submit = pd.read_csv('c:/_data/dacon/bird/sample_submission.csv')
submit['label'] = test_preds
submit.to_csv('c:/_data/dacon/bird/csv/bird6.csv', index=False)
