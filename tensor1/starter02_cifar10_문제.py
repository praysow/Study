# # Question
# #
# # Create a classifier for the CIFAR10 dataset
# # Note that the test will expect it to classify 10 classes and that the input shape should be
# # the native CIFAR size which is 32x32 pixels with 3 bytes color depth

# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import *
# def solution_model():
#     # cifar = tf.keras.datasets.cifar10
#     (x_train,y_train), (x_test,y_test) =  tf.keras.datasets.cifar10.load_data()
#     #2.모델구성
#     model = Sequential()
#     model.add(Conv2D(16,(3,3),input_shape = (32,32,3),activation='swish'))
#     model.add(Conv2D(24,(3,3),activation='swish'))
#     model.add(Conv2D(32,(3,3),activation='swish'))
#     model.add(Flatten())
#     model.add(Dense(10,activation='softmax'))
    
#     #3.컴파일 훈련
#     model.compile(loss='SparseCategorical_Crossentropy',optimizer='adam',metrics='accuracy')
#     model.fit(x_train,y_train,epochs=100,verbose=1,validation_split=0.1) 
    
#     loss = model.evaluate(x_test,y_test)
#     print('loss:',loss[0])
#     print('acc:',loss[1])
    
#     return model



# # Note that you'll need to save your model as a .h5 like this
# # This .h5 will be uploaded to the testing infrastructure
# # and a score will be returned to you
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense

def solution_model():
    # 데이터 불러오기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 모델 구성
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    # 컴파일, 훈련
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.1)

    # 모델 평가
    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy:', accuracy)

    return model

# 모델 생성 및 저장
if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
'''
loss: 3.0625479221343994
accuracy: 0.49950000643730164
'''