# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:03:29 2019

@author: AmP
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def dx(x1, x2):
    return np.array([
            [.02*x1 + .13*x2 - .47*x2**2],
            [-(.07*x2 - .29*x2**2 + .02*x1*x2)]
            ])


def deps(x1, x2):
    return np.deg2rad(-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


def sumsin(x, n):
    return 1/np.sin(x/2)*np.sin((n+1)*x/2)*np.sin(n*x/2)

def sumcos(x, n):
    return 1/np.sin(x/2)*np.sin((n+1)*x/2)*np.cos(n*x/2)


def R(alp):
    return np.array(
            [[np.cos(alp), -np.sin(alp)],
             [np.sin(alp), np.cos(alp)]
             ])

def sumR(alp, n):
    return np.array(
            [[sumcos(alp, n), -sumsin(alp, n)],
             [sumsin(alp, n), sumcos(alp, n)]
             ])


def calc_d(xbar, dx, deps, n):
    return np.linalg.norm(
            R(-n*deps)*xbar - sumR(-deps, n)*dx)



# %% Plot SumSin
n = np.arange(4, 10, 1)
xdeps = np.arange(-4*np.pi, 4*np.pi, .01)
SUMSIN = np.zeros((len(xdeps), len(n)))

for nidx, ni in enumerate(n):
    for didx, depsi in enumerate(xdeps):
        SUMSIN[didx][nidx] = sumsin(depsi, ni)



N, DEPS = np.meshgrid(n, xdeps)
fig = plt.figure('Delta Epsilon')
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(N, DEPS, SUMSIN, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
surf = ax.plot_wireframe(N, DEPS, SUMSIN, rcount=0, ccount=10)

cset = ax.contour(N, DEPS, SUMSIN, zdir='x', offset=4, cmap=cm.coolwarm)
#cset = ax.contour(N, DEPS, SUMSIN, zdir='y', offset=4.1*np.pi, cmap=cm.coolwarm)

ax.set_xlabel('n')
ax.set_ylabel('deps')
ax.set_zlabel('sumsin')

fig.set_size_inches(18.5, 10.5)


# %% Plot D(x1, x2)
class nf(float):
    def __repr__(self):
        s = '{:.1f}'.format(self)  # round to 1 digit
        return '{:.0f}'.format(self) if s[-1] == '0' else s


def plot_contour(X1, X2, D, lines=5):
    X1, X2 = np.meshgrid(X1, X2)
    fig, ax = plt.subplots()
    cs = ax.contourf(X1, X2, D, lines)
    cs = ax.contour(X1, X2, D, lines, colors='k')
    cs.levels = [nf(val) for val in cs.levels]

    if plt.rcParams["text.usetex"]:
        fmt = r'%r'
    else:
        fmt = '%r'
    ax.clabel(cs, cs.levels, inline=1, fontsize=10, fmt=fmt)

    cs.collections[0].set_label('test')
    plt.xlabel('x1')
    plt.ylabel('x2')




xbar = np.array([[2], [2]])
X1 = np.arange(0, 90, 2)
X2 = np.arange(-.5, .5, .01)
D = np.zeros((len(X2), len(X1)))
n = 2
for idx1, x1 in enumerate(X1):
    for idx2, x2 in enumerate(X2):
        D[idx2][idx1] = calc_d(xbar, dx(x1, x2), deps(x1, x2), n)

plot_contour(X1, X2, D)


