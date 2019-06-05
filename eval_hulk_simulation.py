# -*- coding: utf-8 -*-
"""
Created on Tue Jun 04 11:34:15 2019

@author: AmP
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np

from Src.Utils import save_csv


def convert_2d(x1, x2, x3):
    x1_ = np.unique(np.array(x1))
    x2_ = np.unique(np.array(x2))
    x3 = np.array(x3)
    X1, X2 = np.meshgrid(x1_, x2_)
    X3 = np.zeros((len(x2_), len(x1_)))
    idx3 = 0
    for idx1 in range(len(x1_)):
        for idx2 in range(len(x2_)):
            X3[idx2][idx1] = x3[idx3]
            idx3 += 1
    return X1, X2, X3


dic = save_csv.read_csv('Out/190604_simulation_results.csv')

x = dic['x']
dic['x3'] = [xx[0] for xx in x]
dic['x4'] = [xx[1] for xx in x]

# set 1
data1 = {
        'x0': dic['x0'][:len(x)/2],
        'x1': dic['x1'][:len(x)/2],
        'x2': dic['x6'][:len(x)/2],
        'x3': dic['x3'][:len(x)/2],
        'x4': dic['x4'][:len(x)/2],
        'deps': dic['deps'][:len(x)/2],
        'cost': dic['cost'][:len(x)/2]
        }
# set 2
data2 = {
        'x0': dic['x0'][len(x)/2:],
        'x1': dic['x1'][len(x)/2:],
        'x2': dic['x6'][len(x)/2:],
        'x3': dic['x3'][len(x)/2:],
        'x4': dic['x4'][len(x)/2:],
        'deps': dic['deps'][len(x)/2:],
        'cost': dic['cost'][len(x)/2:]
        }


# %% Data 1
x1, x2, x3, x4 = data1['x1'], data1['x2'], data1['x3'], data1['x4']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c='b')
ax.scatter(x1, x2, x4, c='r')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(['x3', 'x4'])


X1, X2, X3 = convert_2d(x1, x2, x3)
surf = ax.plot_wireframe(X1, X2, X3, colors='b')

X1, X2, X4 = convert_2d(x1, x2, x4)
surf = ax.plot_wireframe(X1, X2, X4, colors='r')

# %%
plt.figure('X3 - x00')
plt.contourf(X1, X2, X3, cmap=cm.coolwarm)

plt.figure('X4 - x00')
plt.contourf(X1, X2, X4, cmap=cm.coolwarm)


#%%
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(X1, X2, X3, rstride=8, cstride=8, alpha=0.3)
cset = ax.contour(X1, X2, X3, zdir='z', offset=-15, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='x', offset=60, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='y', offset=.5, cmap=cm.coolwarm)

# %% Data 2
x1, x2, x3, x4 = data2['x1'], data2['x2'], data2['x3'], data2['x4']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c='b')
ax.scatter(x1, x2, x4, c='r')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.legend(['x3', 'x4'])


X1, X2, X3 = convert_2d(x1, x2, x3)
surf = ax.plot_wireframe(X1, X2, X3, colors='b')

X1, X2, X4 = convert_2d(x1, x2, x4)
surf = ax.plot_wireframe(X1, X2, X4, colors='r')


# %%
idx = 21
plt.figure('x1:60')
plt.plot(data1['x2'][:idx], data1['x3'][:idx], 'r-')
plt.plot(data2['x2'][:idx], data2['x3'][:idx], 'r:')

plt.plot(data1['x2'][:idx], data1['x4'][:idx], 'b-')
plt.plot(data2['x2'][:idx], data2['x4'][:idx], 'b:')


#plt.figure('x1:65')
plt.plot(data1['x2'][idx:2*idx], data1['x3'][idx:2*idx], 'r-')
plt.plot(data2['x2'][idx:2*idx], data2['x3'][idx:2*idx], 'r:')

plt.plot(data1['x2'][idx:2*idx], data1['x4'][idx:2*idx], 'b-')
plt.plot(data2['x2'][idx:2*idx], data2['x4'][idx:2*idx], 'b:')

#plt.figure('x1:70')
plt.plot(data1['x2'][2*idx:3*idx], data1['x3'][2*idx:3*idx], 'r-')
plt.plot(data2['x2'][2*idx:3*idx], data2['x3'][2*idx:3*idx], 'r:')

plt.plot(data1['x2'][2*idx:3*idx], data1['x4'][2*idx:3*idx], 'b-')
plt.plot(data2['x2'][2*idx:3*idx], data2['x4'][2*idx:3*idx], 'b:')


plt.show()