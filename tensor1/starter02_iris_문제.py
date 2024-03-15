# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets question
#
# For this task you will train a classifier for Iris flowers using the Iris dataset
# The final layer in your neural network should look like: tf.keras.layers.Dense(3, activation=tf.nn.softmax)
# The input layer will expect data in the shape (4,)
# We've given you some starter code for preprocessing the data
# You'll need to implement the preprocess function for data.map

# import tensorflow as tf
# import tensorflow_datasets as tfds
# data = tfds.load("iris", split='train')

# def preprocess(features):
#     # YOUR CODE HERE
#     # Should return features and one-hot encoded labels
#     return f,l

# def solution_model():
#     train_dataset = data.map(preprocess).batch(10)

#     # YOUR CODE TO TRAIN A MODEL HERE
#     return model


# # Note that you'll need to save your model as a .h5 like this.
# # When you press the Submit and Test button, your saved .h5 model will
# # be sent to the testing infrastructure for scoring
# # and the score will be returned to you.
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import tensorflow as tf
import tensorflow_datasets as tfds

# Iris 데이터셋 로드
data = tfds.load("iris", split='train')

def preprocess(features):
    # 특성과 레이블 추출
    x = features['features']
    y = tf.one_hot(features['label'], depth=3)  # 레이블을 one-hot 인코딩으로 변환
    return x, y

def solution_model():
    # 데이터 전처리 및 배치 설정
    train_dataset = data.map(preprocess).batch(10)

    # 모델 구성
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),  # 입력 레이어
        tf.keras.layers.Dense(16, activation='relu'),  # 은닉 레이어
        tf.keras.layers.Dense(8, activation='relu'),  # 은닉 레이어
        tf.keras.layers.Dense(3, activation='softmax')  # 출력 레이어
    ])

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 훈련
    model.fit(train_dataset, epochs=100)
    
    return model

# # 모델 생성 및 저장
if __name__ == '__main__':
    model = solution_model()
#     model.save("mymodel.h5")

'''
loss: 0.0800 - accuracy: 0.9800
'''