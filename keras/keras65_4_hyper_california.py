import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from keras.models import Model
from keras.layers import *
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV,train_test_split

# 1. 데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr=0.001):
    inputs = Input(shape=(8,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(1, name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='mse', optimizer=optimizer)
    return model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',mode='auto',verbose=1,patience=10)
rlr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=1,factor=0.5)
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
            'node3': node3,
            'callbacks': [es,rlr]}

hyperparameters = create_hyperparameters()
print(hyperparameters)

model = KerasRegressor(build_fn=build_model, verbose=0)
random_search = RandomizedSearchCV(estimator=model, param_distributions=hyperparameters, n_iter=1, n_jobs=16, verbose=1)
random_search.fit(x_train, y_train, epochs=1)
score=random_search.score(x_test,y_test)
print('params',random_search.best_params_)
print('score',score)
print('bestscore',random_search.best_score_)
print('best_estimator',random_search.best_estimator_)

from sklearn.metrics import r2_score
y_predict = random_search.predict(x_test)
mse = r2_score(y_test, y_predict)
print('MSE:', mse)

'''
{'optimizer': 'adam', 'node3': 128, 'node2': 32, 'node1': 32, 'drop': 0.2, 'batch_size': 3, 'activation': 'relu'}
score -1.4858506917953491
bestscore -1.7667654037475586
best_estimator <keras.wrappers.scikit_learn.KerasRegressor object at 0x00000174A8C91700>
MSE: 1.485849904907436
'''