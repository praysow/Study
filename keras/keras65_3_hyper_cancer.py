import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# 1. 데이터
datasets= load_breast_cancer()
x = datasets.data
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=130)

scaler = StandardScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=128, node2=64, node3=32, lr=0.001):
    inputs = Input(shape=(30,), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)

    outputs = Dense(2, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
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
'''
{'optimizer': 'rmsprop', 'node3': 64, 'node2': 16, 'node1': 128, 'drop': 0.4, 'batch_size': 3, 'activation': 'linear'}
score 0.8304093480110168
bestscore 0.7235759556293487
best_estimator <keras.wrappers.scikit_learn.KerasClassifier object at 0x00000240B24E5FA0>
6/6 [==============================] - 0s 600us/step
acc 0.8304093567251462
'''