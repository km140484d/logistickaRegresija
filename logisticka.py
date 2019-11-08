# logisticka

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn


def standardization(x):
    xs = np.copy(x)
    for i in range(1, n):
        xa = np.average(xs[:, i])
        std = np.std(xs[:, i])
        for j in range(m):
            xs[j, i] = (xs[j, i] - xa) / std
    return xs


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


def gradient_loss(x, y):
    alpha_l, alpha, alpha_h = 0.01, 0.02, 0.04
    theta_l, J_l = gradient_descent_j(x, y, alpha_l, 1)
    theta, J = gradient_descent_j(x, y, alpha, 1)
    theta_h, J_h = gradient_descent_j(x, y, alpha_h, 1)
    plt.plot(range(len(J_l)), J_l)
    plt.plot(range(len(J)), J)
    plt.plot(range(len(J_h)), J_h)
    plt.legend(['alpha = 0.01', 'alpha = 0.02', 'alpha = 0.04'], loc='upper right')
    plt.xlabel('iter')
    plt.ylabel('J')
    plt.show()


def gradient_descent_j(x, y, alpha=0.02, flag=0):
    print('gradient_descent')
    theta = np.zeros((n, 1))
    bound = 2e-2
    J = []
    dl = gradient(x, y, theta)
    while np.linalg.norm(dl) > bound:
        theta = theta + alpha * dl
        dl = gradient(x, y, theta)
        if flag:
            dJ = 0
            for i in range(m):
                ht = h(theta, xs[i].T)
                dJ = dJ + y[i] * math.log(ht) + (1 - y[i]) * math.log(1 - ht)
            J.append(-dJ)
    return theta, J


def gradient_descent(x, y):
    theta, J = gradient_descent_j(x, y)
    return theta


# softmax
def delta(x, y, theta):
    m = x.shape[0]
    deltaJ = np.zeros((k, n))
    for r in range(k - 1):
        for i in range(m):
            s = 0
            for j in range(k):
                s = s + math.exp(theta[j].dot(x[i].T))
            deltaJ[r] = deltaJ[r] + ((y[i] == r) - math.exp(theta[r].dot(x[i].T)) / s) * x[i]
    return deltaJ


def gauss(x, my, sigma):
    sigma2 = math.pow(sigma, 2)
    return 1 / math.sqrt(2 * math.pi * sigma2) * math.exp(-math.pow((x - my), 2) / 2 * sigma2)


def gnb(x, my1, sigma1, my0, sigma0):
    invS1, invS0 = np.linalg.inv(sigma1), np.linalg.inv(sigma0)
    return math.exp(0.5 * x.T.dot(invS1).dot(x) - my1.T.dot(invS1).dot(x) + 0.5 * my1.T.dot(invS1).dot(my1)
                    - 0.5 * x.T.dot(invS0).dot(x) + my0.T.dot(invS0).dot(x) - 0.5 * my0.T.dot(invS0).dot(my0))


def plot_conf(conf, reg, train):
    if train == 1:
        print(reg)
        print('conf_train:')
    else:
        print('conf_test:')
    print(conf)
    df_cm = pd.DataFrame(conf, range(k), range(k))
    hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
    bottom, top = hm.get_ylim()
    hm.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


df = pd.read_csv('multiclass_data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
boundary_index = round(df.shape[0] * 0.8)
df = df.sample(frac=1)

y = df['y'].to_numpy()
x = df.iloc[:, 0:6].to_numpy()
m, n, k = x.shape[0], x.shape[1], len(np.unique(y))  # n ukljucuje kolonu sa 1
xs = standardization(x)

# # logisticka regresija
# y0, y1, y2 = np.copy(y), np.copy(y), np.copy(y)
# y0[y0 >= 1], y0[y0 == 0], y0[y0 > 1] = 2, 1, 0
# y1[y1 != 1] = 0
# y2[y2 <= 1], y2[y2 == 2] = 0, 1
# theta0, theta1, theta2 = gradient_descent(xs, y0), gradient_descent(xs, y1), gradient_descent(xs, y2)
# conf_train, conf_test = np.zeros((k, k)), np.zeros((k, k))
# y_guess = np.zeros((m, 1), int)
# for i in range(m):
#     h0, h1, h2 = h(theta0, xs[i].T), h(theta1, xs[i].T), h(theta2, xs[i].T)
#     if h0 > h1 and h0 > h2:
#         y_guess[i] = 0
#     elif h1 > h0 and h1 > h2:
#         y_guess[i] = 1
#     else:
#         y_guess[i] = 2
#     if i < boundary_index:
#         conf_train[y[i], y_guess[i]] = conf_train[y[i], y_guess[i]] + 1
#     else:
#         conf_test[y[i], y_guess[i]] = conf_test[y[i], y_guess[i]] + 1
# plot_conf(conf_train, 'LOGISTIČKA:', 1)
# plot_conf(conf_test, 'LOGISTIČKA:', 0)
# gradient_loss(xs, y1)   # funkcija gubitka u zavisnosti od stope ucenja

# # softmax
# shuffle = np.arange(m)
# row_num = [5, 10, 20]
# row_size = row_num[1]
# for row in row_num:
#     alpha, step, cnt = 0.02, 0, 1000
#     theta_row, J = np.zeros((k, n)), []
#     for i in range(cnt):
#         theta_row = theta_row + alpha * delta(xs[step:min(m, step + row)], y[step:min(m, step + row)], theta_row)
#         dJ = 0
#         for i in range(m):
#             y_guess = 0
#             for j in range(k):
#                 y_guess = y_guess + math.exp(theta_row[j].dot(xs[i].T))
#             dJ = dJ + (theta_row[y[i]].dot(xs[i].T) - math.log(y_guess))
#         J.append(-dJ)
#         step = (step + row) % m
#         if step < row:
#             step = 0
#             np.random.shuffle(shuffle)
#             xs, y = xs[shuffle], y[shuffle]
#     if row == row_size:
#         theta = theta_row
#     plt.plot(range(len(J)), J)
# plt.legend(['šarža = 5', 'šarža = 10', 'šarža = 20'], loc='upper right')
# plt.xlabel('iter')
# plt.ylabel('J')
# plt.show()
# conf_train, conf_test = np.zeros((k, k)), np.zeros((k, k))
# for i in range(m):
#     phi, s = np.zeros((k, 1)), 0
#     for r in range(k):
#         phi[r] = math.exp(theta[r].dot(xs[i].T))
#         s = s + math.exp(theta[r].dot(xs[i].T))
#     phi = phi / s
#     phi_max_index = np.argmax(phi)
#     if i < boundary_index:
#         conf_train[y[i], phi_max_index] = conf_train[y[i], phi_max_index] + 1
#     else:
#         conf_test[y[i], phi_max_index] = conf_test[y[i], phi_max_index] + 1
# plot_conf(conf_train, 'SOFTMAX:', 1)
# plot_conf(conf_test, 'SOFTMAX:', 0)

# GDA - Gausovska diskriminantna analiza
xs = xs[:, 1:]
xs = np.c_[xs, y]
n = n - 1  # nema potrebe vise za kolonom sa 1
xs0, xs1, xs2 = xs[np.where(xs[:, n] == 0)], xs[np.where(xs[:, n] == 1)], xs[np.where(xs[:, n] == 2)]
xs0, xs1, xs2 = xs0[:, :-1], xs1[:, :-1], xs2[:, :-1]
x_sep = [xs0, xs1, xs2]
my, sigma = np.zeros((k, n)), np.zeros((k, n))
# racunanje my-matematicko ocekivanje, sigma-standardna devijansa
for i in range(k):
    for j in range(n):
        my[i, j] = np.mean(x_sep[i][:, j])
        sigma[i, j] = np.sd(x_sep[i][:, j])
conf_train, conf_test = np.zeros((k, k)), np.zeros((k, k))
for i in range(m):
    gm, p = np.zeros((k, n)), np.zeros(k)  # gauss matrix
    total = 0
    for l in range(k):
        for j in range(n):
            gm[l, j] = gauss(xs[i, j], my[l, j], sigma[l, j])
        p[l] = np.prod(gm[l])
        total = total + p[l]
    p = p / total
    if i < boundary_index:
        conf_train[y[i], np.argmax(p)] = conf_train[y[i], np.argmax(p)] + 1
    else:
        conf_test[y[i], np.argmax(p)] = conf_test[y[i], np.argmax(p)] + 1
plot_conf(conf_train, 'GDA:', 1)
plot_conf(conf_test, 'GDA:', 0)

# # GNB - Naivni Bayes
# MY0 = np.ones((5, xs0.shape[0]))
# MY1 = np.ones((5, xs1.shape[0]))
# MY2 = np.ones((5, xs2.shape[0]))
# for j in range(n):
#     MY0[j] = my[0, j]
#     MY1[j] = my[1, j]
#     MY2[j] = my[2, j]
# # print('my', my)
# # print()
# # print('MY0', MY0)
# SIGMA0 = 1 / (xs0.shape[0] - 1) * (xs0.T - MY0).dot((xs0.T - MY0).T)
# print('SIGMA0', SIGMA0)
# SIGMA1 = 1 / (xs1.shape[0] - 1) * (xs1.T - MY1).dot((xs1.T - MY1).T)
# print('SIGMA1', SIGMA1)
# SIGMA2 = 1 / (xs2.shape[0] - 1) * (xs2.T - MY2).dot((xs2.T - MY2).T)
# print('SIGMA2', SIGMA2)
#
# conf = np.zeros((k, k))
# xs = xs[:, :-1]     # izbacivanje kolone sa 1
# for i in range(m):
#     p = np.zeros(k)
#     p[0] = 1 / (1 + gnb(xs[i].T, my[1].T, SIGMA1, my[0].T, SIGMA0) + gnb(xs[i].T, my[2].T, SIGMA2, my[0].T, SIGMA0))
#     p[1] = 1 / (1 + gnb(xs[i].T, my[0].T, SIGMA0, my[1].T, SIGMA1) + gnb(xs[i].T, my[2].T, SIGMA2, my[1].T, SIGMA1))
#     p[2] = 1 / (1 + gnb(xs[i].T, my[0].T, SIGMA0, my[2].T, SIGMA2) + gnb(xs[i].T, my[1].T, SIGMA1, my[2].T, SIGMA2))
#     conf[y[i], np.argmax(p)] = conf[y[i], np.argmax(p)] + 1
# print('conf [gnb]: ', conf)
# df_cm = pd.DataFrame(conf, range(k), range(k))
# hm = sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
# bottom, top = hm.get_ylim()
# hm.set_ylim(bottom + 0.5, top - 0.5)
# # plt.show()
