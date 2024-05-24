# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.arange(-5, 5, 0.1)
# a = 1
# alpha = 1
# scale = 1
# def selu(x, alpha = 1, scale = 1):
#     return np.where(x <= 0, scale * alpha * (np.exp(x) -1 ), scale * x)
#
# # selu = lambda x : np.where(x <= 0, scale * alpha * (np.exp(x) -1 ), scale * x)
# y = selu(x)
#
# plt.plot(x, y)
# plt.grid()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * (x * (x > 0) + alpha * (np.exp(x) - 1) * (x <= 0))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()
