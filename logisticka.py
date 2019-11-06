# logisticka

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def h(theta, x):
    gamma = -theta.T.dot(x)
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


# softmax
def delta(x, y, theta):
    m = x.shape[0]
    deltaJ = np.zeros((k, n))
    for r in range(k-1):
        for i in range(m):
            s = 0
            for j in range(k):
                s = s + math.exp(theta[j].dot(x[i].T))
            deltaJ[r] = deltaJ[r] + ((y[i] == r) - math.exp(theta[r].dot(x[i].T))/s) * x[i]    #
    return deltaJ


df = pd.read_csv('multiclass_data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
df = df.sample(frac=1)

y = df['y'].to_numpy()
x = df.iloc[:, 0:6].to_numpy()
m, n, k = x.shape[0], x.shape[1], len(np.unique(y))     # n ukljucuje kolonu sa 1
xs = np.copy(x)
# standardizacija
for i in range(1, 6):
    xa = np.average(xs[:, i])
    std = np.std(xs[:, i])
    for j in range(m):
        xs[j, i] = (xs[j, i] - xa) / std

y0, y1, y2 = np.copy(y), np.copy(y), np.copy(y)
y0[y0 >= 1], y0[y0 == 0], y0[y0 > 1] = 2, 1, 0
y1[y1 != 1] = 0
y2[y2 <= 1], y2[y2 == 2] = 0, 1
theta0, theta1, theta2 = gradient_descent(xs, y0), gradient_descent(xs, y1), gradient_descent(xs, y2)

# konfuziona matrica
conf = np.zeros((k, k))
y_guess = np.zeros((m, 1), int)
for i in range(m):
    h0 = h(theta0, xs[i].T)
    h1 = h(theta1, xs[i].T)
    h2 = h(theta2, xs[i].T)
    if h0 > h1 and h0 > h2:
        y_guess[i] = 0
    elif h1 > h0 and h1 > h2:
        y_guess[i] = 1
    else:
        y_guess[i] = 2
    conf[y[i], y_guess[i]] = conf[y[i], y_guess[i]] + 1

print('conf [log]: ', conf)

# softmax
alpha, step, row_num, cnt = 0.02, 0, 10, 1000
theta = np.zeros((k, n))
shuffle = np.arange(m)
for i in range(cnt):
    theta = theta + alpha*delta(xs[step:min(m, step+row_num)], y[step:min(m, step+row_num)], theta)
    step = (step + row_num) % m
    if step < row_num:
        step = 0
        np.random.shuffle(shuffle)
        xs = xs[shuffle]
        y = y[shuffle]
conf = np.zeros((k, k))
for i in range(m):
    phi = np.zeros((k, 1))
    s = 0
    for r in range(k):
        s = s + math.exp(theta[r].dot(xs[i].T))
    for r in range(k):
        phi[r] = math.exp(theta[r].dot(xs[i].T))/s
    phi_max_index = np.argmax(phi)
    conf[y[i], phi_max_index] = conf[y[i], phi_max_index] + 1
print('conf [softmax]: ', conf)
