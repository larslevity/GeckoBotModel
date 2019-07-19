#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:04:25 2019

@author: ls
"""

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model
from Src.Utils import save


def gnerate_ptrn(X, n_cycles, half=False):
    ptrn = []
    for n in range(n_cycles):
        p = X
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
        ptrn.append([[p[5], p[6], p[7], p[8], p[9]], [0, 1, 1, 0]])
    if half:
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
    return ptrn


def mirror(p):
    return [p[1], p[0], -p[2], p[4], p[3]]


def gnerate_ptrn_symmetric(X, n_cycles, half=False):
    ptrn = []
    for n in range(n_cycles):
        p = X
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
        ptrn.append([[p[1], p[0], -p[2], p[4], p[3]], [0, 1, 1, 0]])
    if half:
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
    return ptrn



cat = 'curve'
f_len = 10
f_ang = 100
f_ori = 10
f_weights = [f_len, f_ori, f_ang]



alp_opt = [97, 28, -98, 116, 17, 79, 0.1, -84, 67, .1]


x, marks, f = model.set_initial_pose(
    alp_opt[:5], 90, (0, 0), len_leg=1, len_tor=1.2)
initial_pose = pf.GeckoBotPose(x, marks, f)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)
if cat == 'straight':
    ptrn = gnerate_ptrn_symmetric(alp_opt, 2, half=True)
if cat == 'curve':
    ptrn = gnerate_ptrn(alp_opt, 2, half=True)
for ref in ptrn:
    x, marks, f, constraint, cost = model.predict_next_pose(
            ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
    gait.append_pose(
            pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))

gait.plot_gait()
gait.plot_markers([1])
plt.axis('off')
plt.savefig('Out/opt_ref/'+cat+'/{}.png'.format('test'),
            transparent=False, dpi=150)
plt.show()
plt.close('GeckoBotGait')

# %% SAVE AS TIKZ
gait.plot_markers(1)
plt.axis('off')
gait_str = gait.get_tikz_repr()
save.save_plt_as_tikz('Out/opt_ref/'+cat+'/{}.tex'.format('test'),
                      gait_str)
plt.close('GeckoBotGait')