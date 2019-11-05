# logisticka

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def h(theta, x):
    return 1 / (1 + math.exp(-theta.T.dot(x)))


# gradijentni spust, maksimizacija verodostojnosti
def gradient(x, y, theta):
    gl = np.zeros((n, 1))
    for i in range(m):
        h_theta = h(theta, x[i].T)
        print(i, h_theta)
        for j in range(n):
            gl[j] = gl[j] + (y[i] - h_theta) * x[i, j]
    return gl


def gradient_descent(x, y):
    print()
    print('gradient_descent')
    alpha = 0.01    # konstanta ucenja
    theta = np.zeros((n, 1))
    bound = 1e-2
    dl = gradient(x, y, theta)
    while np.linalg.norm(dl) > bound:
        theta = theta + alpha*dl
        dl = gradient(x, y, theta)
    return theta


df = pd.read_csv('multiclass_data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
# df = df.sample(frac=1)

x = df.iloc[:, 0:6].to_numpy()
y = df['y'].to_numpy()

m, n = x.shape[0], x.shape[1]

print(m, ', ', n)

y0, y1, y2 = np.copy(y), np.copy(y), np.copy(y)
y0[y0 >= 1], y0[y0 == 0], y0[y0 > 1] = 2, 1, 0
y1[y1 != 1] = 0
y2[y2 <= 1], y2[y2 == 2] = 0, 1
theta0 = gradient_descent(x, y0)
print('y0 ', y0)
print('theta0: ', theta0)
print()

# y1[y1 != 1] = 0
# theta1 = gradient_descent(x, y1)
# print('y1', y1)
# print('theta1: ', theta1)
# print()
#
# y2[y2 < 2] = 0
# y2[y2 == 2] = 1
# print('y2', y2)

# # konfuziona matrica
# conf = np.zeros((n-1, n-1))
# print(conf)

# # print(h(np.zeros(n), x))
# theta = np.array([-0.3, -3.462, -0.04645, -1.24045, -0.4613, -5.925])
# x = np.array([1.00e+00, 1.42e+01, 3.06e+00, 5.64e+00, 3.92e+00, 1.07e+03])
# # h(theta, x)
# print('theta', theta)
# print('x: ', x.T)
# print(h(theta, x.T))
# # print(1 / (1 + round(math.exp(-theta.dot(x.T)), 10)))
#
# # logistička
#
# theta = np.zeros((n, 1))
# print(theta)


theta, x = np.array([1, 2, 3]), np.array([4, 5, 6])
1 / (1 + round(math.exp(-theta.T.dot(x)), 10))

