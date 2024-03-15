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
# You must use the Submit and Test model button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Getting Started Question
#
# Given this data, train a neural network to match the xs to the ys
# So that a predictor for a new value of X will give a float value
# very close to the desired answer
# i.e. print(model.predict([10.0])) would give a satisfactory result
# The test infrastructure expects a trained model that accepts
# an input shape of [1]

# import numpy as np
# from keras.models import Sequential
# from keras.layers import *

# def solution_model():
#     xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
#     ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

#     model = Sequential()
#     model.add(Dense(1,input_shape = (1,)))
#     model.add(Dense(1))

#     model.compile(loss='mse',optimizer='adam')
#     model.fit(xs,ys,epochs=10,verbose=1)
    
#     loss = model.evaluate(xs,ys)
#     result = model.predict(10.0)
#     print('loss',loss)
#     print('result',result)
#     return model
# # solution_model() 함수 호출
# model = solution_model()

# # 결과 출력
# print('Loss:', model.evaluate(xs, ys))
# print('Prediction for 10.0:', model.predict(np.array([10.0])))


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
# if __name__ == '__main__':
#     model = solution_model()
#     model.save("mymodel.h5")

import numpy as np
from keras.models import Sequential
from keras.layers import *

def solution_model():
    # xs와 ys 정의
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(xs, ys, epochs=1000, verbose=1)
    
    # evaluate 및 predict에 xs와 ys 사용
    loss = model.evaluate(xs, ys)
    result = model.predict(np.array([10.0]))
    print('loss', loss)
    print('result', result)
    return model

if __name__ == '__main__':
    model = solution_model()
    # model.save("mymodel.h5")
'''
loss 0.00746202515438199
result [[10.644938]]
'''