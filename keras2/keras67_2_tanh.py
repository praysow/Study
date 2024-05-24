import numpy as np
import matplotlib.pyplot as plt

tanh = lambda x: np.tanh(x)

x= np.arange(-5,5,0.1)
print(tanh(x))
print(len(x))

y = tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()