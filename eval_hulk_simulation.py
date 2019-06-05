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
idx = int(len(x)/2)
data1 = {
        'x0': dic['x0'][:idx],
        'x1': dic['x1'][:idx],
        'x2': dic['x6'][:idx],
        'x3': dic['x3'][:idx],
        'x4': dic['x4'][:idx],
        'deps': dic['deps'][:idx],
        'cost': dic['cost'][:idx]
        }
# set 2
data2 = {
        'x0': dic['x0'][idx:],
        'x1': dic['x1'][idx:],
        'x2': dic['x6'][idx:],
        'x3': dic['x3'][idx:],
        'x4': dic['x4'][idx:],
        'deps': dic['deps'][idx:],
        'cost': dic['cost'][idx:]
        }


# %% Data 1
x1, x2, x3, x4 = data1['x1'], data1['x2'], data1['x3'], data1['x4']

X1, X2, X3 = convert_2d(x1, x2, x3)
X1, X2, X4 = convert_2d(x1, x2, x4)


# %%
fig = plt.figure('X3-x0_0')
ax = fig.gca(projection='3d')


surf = ax.plot_wireframe(X1, X2, X3, colors='b')
ax.scatter(x1, x2, x3, c='b')
ax.plot_surface(X1, X2, X3, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='z', offset=-15, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='x', offset=60, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='y', offset=.5, cmap=cm.coolwarm)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')


# %%
fig = plt.figure('X4-x0_0')
ax = fig.gca(projection='3d')


surf = ax.plot_wireframe(X1, X2, X4, colors='r')
ax.scatter(x1, x2, x4, c='r')
ax.plot_surface(X1, X2, X4, rstride=8, cstride=8, alpha=0.1, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='z', offset=17, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='x', offset=60, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='y', offset=.5, cmap=cm.coolwarm)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x4')


# %% Data 2
x1, x2, x3, x4 = data2['x1'], data2['x2'], data2['x3'], data2['x4']
X1, X2, X3 = convert_2d(x1, x2, x3)
X1, X2, X4 = convert_2d(x1, x2, x4)

# %%
fig = plt.figure('X3-x0_1')
ax = fig.gca(projection='3d')


surf = ax.plot_wireframe(X1, X2, X3, colors='b')
ax.scatter(x1, x2, x3, c='b')
ax.plot_surface(X1, X2, X3, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='z', offset=-15, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='x', offset=60, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X3, zdir='y', offset=.5, cmap=cm.coolwarm)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')


# %%
fig = plt.figure('X4-x0_1')
ax = fig.gca(projection='3d')


surf = ax.plot_wireframe(X1, X2, X4, colors='r')
ax.scatter(x1, x2, x4, c='r')
ax.plot_surface(X1, X2, X4, rstride=8, cstride=8, alpha=0.1, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='z', offset=17, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='x', offset=60, cmap=cm.coolwarm)
cset = ax.contour(X1, X2, X4, zdir='y', offset=.5, cmap=cm.coolwarm)


ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x4')


# %%


plt.figure('X3-X4 - x0_0 - 2D')
plt.title('x3, x4 for varying x1, x2 -- x0_0')

idx = 21
lx1 = len(np.unique(np.array(x1)))
for jdx in range(lx1):
    d = .8
    col2 = (d-(jdx*(d)/lx1), d-(jdx*(d)/lx1), 1)
    plt.plot(data1['x2'][jdx*idx:(jdx+1)*idx], data1['x3'][jdx*idx:(jdx+1)*idx], '-', color=col2)
#    plt.plot(data2['x2'][jdx*idx:(jdx+1)*idx], data2['x3'][jdx*idx:(jdx+1)*idx], ':', color=col2)

for jdx in range(lx1):
    d = .8
    col1 = (1, d-(jdx*(d)/lx1), d-(jdx*(d)/lx1))
    plt.plot(data1['x2'][jdx*idx:(jdx+1)*idx], data1['x4'][jdx*idx:(jdx+1)*idx], '-', color=col1)
#    plt.plot(data2['x2'][jdx*idx:(jdx+1)*idx], data2['x4'][jdx*idx:(jdx+1)*idx], ':', color=col1)

plt.legend(['x3 for x1={}'.format(int(60 + 10*x)) for x in range(lx1)]
            + ['x4 for x1={}'.format(int(60 + 10*x)) for x in range(lx1)])

plt.xlabel('x2')
plt.ylabel('x3, x4')
plt.grid()

# %%


plt.figure('X3-X4 - x0_1 - 2D')
plt.title('x3, x4 for varying x1, x2 -- x0_1')

idx = 21
lx1 = len(np.unique(np.array(x1)))
for jdx in range(lx1):
    d = .8
    col2 = (d-(jdx*(d)/lx1), d-(jdx*(d)/lx1), 1)
    plt.plot(data2['x2'][jdx*idx:(jdx+1)*idx], data2['x3'][jdx*idx:(jdx+1)*idx], '-', color=col2)

for jdx in range(lx1):
    d = .8
    col1 = (1, d-(jdx*(d)/lx1), d-(jdx*(d)/lx1))
    plt.plot(data2['x2'][jdx*idx:(jdx+1)*idx], data2['x4'][jdx*idx:(jdx+1)*idx], '-', color=col1)

plt.legend(['x3 for x1={}'.format(int(60 + 10*x)) for x in range(lx1)]
            + ['x4 for x1={}'.format(int(60 + 10*x)) for x in range(lx1)])

plt.xlabel('x2')
plt.ylabel('x3, x4')
plt.grid()


plt.show()