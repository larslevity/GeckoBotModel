# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 19:03:29 2019

@author: AmP
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))


from Src.TrajectoryPlanner import optimal_planner as optplanner
from Src.Utils import save
from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model
from Src import calibration


# %% Plot SumSin
#n = np.arange(4, 10, 1)
#xdeps = np.arange(-4*np.pi, 4*np.pi, .01)
#SUMSIN = np.zeros((len(xdeps), len(n)))
#
#for nidx, ni in enumerate(n):
#    for didx, depsi in enumerate(xdeps):
#        SUMSIN[didx][nidx] = optplanner.sumsin(depsi, ni)
#
#
#N, DEPS = np.meshgrid(n, xdeps)
#fig = plt.figure('Delta Epsilon')
#ax = fig.gca(projection='3d')
#surf = ax.plot_wireframe(N, DEPS, SUMSIN, rcount=0, ccount=10)
#cset = ax.contour(N, DEPS, SUMSIN, zdir='x', offset=4, cmap=cm.coolwarm)
#
#ax.set_xlabel('n')
#ax.set_ylabel('deps')
#ax.set_zlabel('sumsin')

# fig.set_size_inches(18.5, 10.5)

# %% Plot D(x1, x2)


class nf(float):
    def __repr__(self):
        s = '{:.1f}'.format(self)  # round to 1 digit
        return '{:.0f}'.format(self) if s[-1] == '0' else s


def plot_contour(X1, X2, D, lines=2):
    X1, X2 = np.meshgrid(X1, X2)
    fig, ax = plt.subplots()
    cs = ax.contourf(X2, X1, D, lines, cmap='RdYlGn_r')
    cs = ax.contour(X2, X1, D, lines, colors='k')
    cs.levels = [nf(val) for val in cs.levels]

    if plt.rcParams["text.usetex"]:
        fmt = r'%r'
    else:
        fmt = '%r'
    ax.clabel(cs, cs.levels, inline=1, fontsize=10, fmt=fmt)

#    cs.collections[0].set_label('test')
    ax.set_xlabel('$q_2$ (1)', fontsize=10)
    ax.set_ylabel('$q_1$ ($^\\circ$)', fontsize=10)

    plt.xticks([-.5, 0, .5])
    plt.yticks([50, 70, 90])
#    ax.xaxis.set_label_coords(1, -.025)
#    ax.yaxis.set_label_coords(-0.025, 1)


#xbar = np.array([[2], [2]])
#X1 = np.arange(0, 90, 2)
#X2 = np.arange(-.5, .5, .01)
#D = np.zeros((len(X2), len(X1)))
#n = 2
#for idx1, x1 in enumerate(X1):
#    for idx2, x2 in enumerate(X2):
#        D[idx2][idx1] = optplanner.calc_d(xbar, optplanner.dx(x1, x2),
#                                          optplanner.deps(x1, x2), n)
#
#plot_contour(X1, X2, D)
#kwargs = {'extra_axis_parameters': {'font=\\small'}}
#save.save_plt_as_tikz('Out/opt_pathplanner/dist_test.tex', **kwargs)

# %% Create Case Study:

xref = [20, 35]  # global cos
eps = 90
p1 = (0, 0)
n = 4


xbar = optplanner.xbar(xref, p1, eps)
print('xbar:', xbar)
X1 = np.arange(50, 90.01, 2)
X2 = np.arange(-.5, .501, .01)
D = np.zeros((len(X2), len(X1)))
dmin = {'val': 1e16, 'x1': None, 'x2': None}
for idx1, x1 in enumerate(X1):
    for idx2, x2 in enumerate(X2):
        d = optplanner.calc_d(xbar, optplanner.dx(x1, x2),
                              optplanner.deps(x1, x2), n)
        if d < dmin['val']:
            dmin = {'val': d, 'x1': x1, 'x2': x2}
        D[idx2][idx1] = d

x1_opt, x2_opt = dmin['x1'], dmin['x2']
plot_contour(X1, X2, D, lines=8)
plt.plot(x2_opt, x1_opt, marker='o', color='purple', markersize=12,
         markeredgecolor='k')
kwargs = {'extra_axis_parameters': {'font=\\small'}}
save.save_plt_as_tikz('Out/opt_pathplanner/dist_n_{}.tex'.format(n),
                      **kwargs)


# %% Gecko

x1, x2 = x1_opt, x2_opt  # optimal gait
feet0 = [1, 0, 0, 1]  # feet states
alpha0 = [90, 0, -90, 90, 0]


weight = [89, 10, 5.9]   # [f_l, f_o, f_a]
len_leg, len_tor = calibration.get_len('vS11')

# alpha = [90, 0, -90, 90, 0]

x, (mx, my), f = model.set_initial_pose(alpha0, eps, p1, len_leg=len_leg,
                                        len_tor=len_tor)
initial_pose = pf.GeckoBotPose(x, (mx, my), f)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)


for cyc in range(n):
    for sign in [1, -1]:
        act_pose = gait.poses[-1]
        x, y = act_pose.markers
        # generate ref
        x1_ = x1*sign
        feet = [not(f) for f in feet0] if sign == 1 else feet0
        alpha = optplanner.alpha(x1_, x2, feet)
        print(x1_, x2, alpha, feet)
        # predict
        predicted_pose = model.predict_next_pose(
                        [alpha, feet], act_pose.x, (x, y), weight,
                        len_leg=len_leg, len_tor=len_tor)
        predicted_pose = pf.GeckoBotPose(*predicted_pose)
        gait.append_pose(predicted_pose)
        # switch feet

# Plot
gait.plot_gait(reverse_col=1)
gait.plot_markers(1)
#    gait.plot_com()
plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)
plt.axis('off')

# plt.savefig('Out/opt_pathplanner/gait.png', transparent=False,
#            dpi=300)
plt.show()
plt.close('GeckoBotGait')

# %% Tikz Pic

gait.plot_markers(1)
plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)
plt.plot(p1[0], p1[1], marker='o', color='red', markersize=12)
plt.axis('off')
gait_str = gait.get_tikz_repr(reverse_col=1, linewidth='.7mm')
save.save_plt_as_tikz('Out/opt_pathplanner/gait_n_{}.tex'.format(n),
                      gait_str, scope='scale=.1')
