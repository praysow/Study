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

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x = x.reshape(70000, 28*28)

scaler = StandardScaler()
x = scaler.fit_transform(x)

n_components_list = [154, 311, 713, 784]
results = {}
from keras.callbacks import EarlyStopping,ModelCheckpoint
for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    print(f"Explained Variance Ratio for {n_components} components:", cumsum[-1])

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.9, random_state=3, shuffle=True, stratify=y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Dense(50, input_shape=(n_components,), activation='relu'))
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
    results[n_components] = {
        "loss": result[0],
        "accuracy": result[1],
        "elapsed_time": round(end_time - start_time)
    }

for n_components, result in results.items():
    print(f"Results for {n_components} components:")
    print("Loss:", result["loss"])
    print("Accuracy:", result["accuracy"])
    print("Elapsed Time:", result["elapsed_time"])
