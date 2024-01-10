import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.animation import FuncAnimation

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=30, activation='linear'))
# Add more layers if needed based on the complexity of the problem

# 3. 컴파일 및 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')

# Function to update the plot during training
def update_plot(epoch, logs):
    loss_history.append(logs['loss'])
    x_values.append(epoch)
    line.set_data(x_values, loss_history)
    ax.relim()
    ax.autoscale_view()

# Set up real-time plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Real-time Training Loss')

loss_history = []
x_values = []

# Function to update the plot during training
def update_plot(epoch, logs):
    loss_history.append(logs['loss'])
    x_values.append(epoch)
    line.set_data(x_values, loss_history)
    ax.relim()
    ax.autoscale_view()

animation = FuncAnimation(fig, update_plot, frames=range(500), interval=200, fargs=({},))

# Start training with animation
hist = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.3, verbose=2, callbacks=[es, animation])

# 4. 결과 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

# Evaluation metrics
r2 = r2_score(y_test, y_predict)
print("로스:", loss)
print("R2 score:", r2)

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse = RMSE(y_test, y_predict)
print("RMSE:", rmse)

plt.show()

