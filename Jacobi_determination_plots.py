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
    return [.02*x1 + .13*x2 - .47*x2**2,
            -(.07*x2 - .29*x2**2 + .02*x1*x2)]


def deps(x1, x2):
    return (-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


def sumsin(x, n):
    return 1/np.sin(x/2)*np.sin((n+1)*x/2)*np.sin(n*x/2)

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

