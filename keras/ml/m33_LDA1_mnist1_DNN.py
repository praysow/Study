from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import time
from keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x = x.reshape(70000, 28*28)

lda = LinearDiscriminantAnalysis(n_components=1)
x= lda.fit_transform(x,y)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
from keras.callbacks import EarlyStopping,ModelCheckpoint

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3, shuffle=True, stratify=y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(50, input_shape=(1,), activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='..\_data\_save\MCP\keras25_MCP19.hdf5'
)

start_time = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, mcp])
end_time = time.time()

result = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print("loss",result)
'''
oss [1.4434468746185303, 0.4327142834663391]
'''