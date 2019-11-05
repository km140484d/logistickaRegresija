# logisticka

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def h(theta, x):
    return 1 / (1 + math.exp(-theta.T.dot(x)))


df = pd.read_csv('multiclass_data.csv', header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
df.insert(0, 'one', 1)
df = df.sample(frac=1)

boundary_index = round(df.shape[0] * 0.8)  # uzeto je da je 80% skup za treniranje, a 20% skup za testiranje
X = df.iloc[:, 0:6].to_numpy()
Y = df['y'].to_numpy()

# logistička


