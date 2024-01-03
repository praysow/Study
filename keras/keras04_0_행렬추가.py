import numpy as np

x1 = np.array([1,2,3])

print("x1 :", x1.shape) # x1 : (3,)

x2 = np.array([[1,2,3]])

print("x2 :", x2.shape) # (1, 3)

x3= np.array([[1,2],[3,4]])

print("x3 :", x3.shape) # (2, 2)

x4= np.array([[1,2],[3,4],[5,6]])

print("x4 :", x4.shape) # (3, 2)

x5= np.array([[[1,2],[3,4],[5,6]]])

print("x5 :", x5.shape) # (1, 3, 2)

x6= np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

print("x6 :", x6.shape) # (2, 2, 2)

x7= np.array([[[[1,2,3,4,5],[6,7,8,9,10]]]])

print("x7 :", x7.shape) # (1, 1, 2, 5)

x8 = np.array([[1,2,3],[4,5,6]])

print("x8 :", x8.shape) # (2, 3)

x9 = np.array([[[[1]]],[[[2]]]])

print("x9 :", x9.shape) # (2, 1, 1, 1)
