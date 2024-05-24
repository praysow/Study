import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)

# def reaky_relu(x):
#     return np.maximum(0.1*x,x)

reaky_relu = lambda x: np.maximum(0.1*x,x)

y = reaky_relu(x)

plt.plot(x,y)
plt.grid()
plt.show()