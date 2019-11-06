# logisticka

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# provereno
def h(theta, x):
    gamma = -theta.T.dot(x)
    # if gamma > 500:
    #     return 0
    # else:
    return 1 / (1 + math.exp(gamma))


# sarzni gradijentni spust, maksimizacija verodostojnosti
def gradient(x, y, theta):
    gl = np.zeros((n, 1))
    for i in range(m):
        h_theta = h(theta, x[i].T)
        for j in range(n):
            gl[j] = gl[j] + (y[i] - h_theta) * x[i, j]
    return gl


def gradient_descent(x, y):
    print('gradient_descent')
    alpha = 0.01  # konstanta ucenja
    theta = np.zeros((n, 1))
    bound = 2e-2
    dl = gradient(x, y, theta)
    while np.linalg.norm(dl) > bound:
        theta = theta + alpha * dl
        dl = gradient(x, y, theta)
    return theta


df = pd.read_csv('multiclass_data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
# df = df.sample(frac=1)

x = df.iloc[:, 0:6].to_numpy()
m, n = x.shape[0], x.shape[1]
xs = np.copy(x)
# standardizacija
for i in range(1, 6):
    xa = np.average(xs[:, i])
    std = np.std(xs[:, i])
    for j in range(m):
        xs[j, i] = (xs[j, i] - xa) / std
y = df['y'].to_numpy()

# print(m, ', ', n)


y0, y1, y2 = np.copy(y), np.copy(y), np.copy(y)
y0[y0 >= 1], y0[y0 == 0], y0[y0 > 1] = 2, 1, 0
y1[y1 != 1] = 0
y2[y2 <= 1], y2[y2 == 2] = 0, 1
theta0 = gradient_descent(xs, y0)
theta1 = gradient_descent(xs, y1)
theta2 = gradient_descent(xs, y2)
print(theta0, theta1, theta2)

# konfuziona matrica
conf = np.zeros((len(np.unique(y)), len(np.unique(y))))
y_guess = np.zeros((m, 1), int)
for i in range(1, m):
    h0 = h(theta0, xs[i].T)
    h1 = h(theta1, xs[i].T)
    h2 = h(theta2, xs[i].T)
    if h0 > h1 and h0 > h2:
        y_guess[i] = 0
    elif h1 > h0 and h1 > h2:
        y_guess[i] = 1
    else:
        y_guess[i] = 2
    # print('y[i]', y[i], 'y_guess[i]', y_guess[i])
    conf[y[i], y_guess[i]] = conf[y[i], y_guess[i]] + 1

print('conf', conf)
