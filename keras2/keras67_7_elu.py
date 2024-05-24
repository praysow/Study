import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
a = 1

# def elu(x):
    # return x*(x>0) + (a * (np.exp(x) - 1)) * (x<=0)

elu = lambda x : x*(x>0) + (a * (np.exp(x) - 1)) * (x<=0)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()
