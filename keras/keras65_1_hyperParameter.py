import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255.
x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255.

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr=0.001):
    inputs = Input(shape=(28 * 28), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    return model

def create_hyperparameters():
    batchs = [1, 2, 3, 4, 5]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    return {'batch_size': batchs,
            'optimizer': optimizers,
            'drop': dropouts,
            'activation': activations,
            'node1': node1,
            'node2': node2,
            'node3': node3}

hyperparameters = create_hyperparameters()
print(hyperparameters)

model = KerasClassifier(build_fn=build_model, verbose=0)
random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=1, n_jobs=16, verbose=1)
random_search.fit(x_train, y_train, epochs=3)
score=random_search.score(x_test,y_test)
print(random_search.best_params_)
print('score',score)
print('bestscore',random_search.best_score_)
print('best_estimator',random_search.best_estimator_)

from sklearn.metrics import accuracy_score
y_predict = random_search.predict(x_test)
print('acc',accuracy_score(y_test,y_predict))