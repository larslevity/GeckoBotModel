#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:37:00 2020

@author: ls
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import matplotlib.animation as animation

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model



alp = [0, 90, 90, 0, 90]
eps = 90
ell = [9.3, 9.3, 11.2, 9.3, 9.3]
p1 = (0, 0)

# REF
ref1 = [[20, 10, -90, 30, 120], [1, 0, 0, 1]]
ref2 = [[80, 100, 120, 0, 100], [0, 1, 1, 0]]


x, marks, f = model.set_initial_pose(
        alp, eps, p1, len_leg=ell[0], len_tor=ell[2])


initial_marks = marks

initial_pose = pf.GeckoBotPose(x, marks, f)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)



[f_l, f_o, f_a] = [89, 10 ,5.9]
dev_ang = 100
blow = .9
bup = 1.1
n_limbs = 5
n_foot = 4


class Memory(object):
    def __init__(self):
        self.val = 0
        
last_obj = Memory()
X = Memory()
X.val = []
Marks = Memory
Marks.val = []
Stress = Memory()
Stress.val = []

def midx(fidx):
    mapped = [0, 2, 3, 5]
    return mapped[fidx]


def predict_next_pose(reference, x_init, markers_init,
                      w=[f_l, f_o, f_a],
                      len_leg=9.3, len_tor=11.2,
                      dev_ang=dev_ang, bounds=(blow, bup), gait=None):
    blow, bup = bounds
    f_len, f_ori, f_ang = w
    ell_nominal = (len_leg, len_leg, len_tor, len_leg, len_leg)

    alpref_, f = reference
    alpref = model._check_alpha(alpref_)
    alp_init, eps_init = x_init[0:n_limbs], x_init[-1]
    phi_init = model._calc_phi(alp_init, eps_init)

    # Initial guess
    x0 = x_init


    def objective(x):
        alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
        phi = model._calc_phi(alp, eps)
        obj_ori, obj_len, obj_ang = 0, 0, 0
        for idx in range(n_foot):
            if f[idx]:
                # dphi = calc_difference(phi[idx], phi_init[idx])
                dphi = phi[idx] - phi_init[idx]
                obj_ori = (obj_ori + dphi**2)
        for idx in range(n_limbs):
            obj_ang = obj_ang + (alpref[idx]-alp[idx])**2
        for idx in range(n_limbs):
            obj_len = obj_len + (ell[idx]-ell_nominal[idx])**2
        objective = (f_ori*np.sqrt(obj_ori) + f_ang*np.sqrt(obj_ang) +
                     f_len*np.sqrt(obj_len))

        if gait and abs(last_obj.val- objective) > .01:
            marks = model._calc_coords(x, markers_init, f)
            for fidx, fi in enumerate(f):
                if fi == 1:
                    marks[0][midx(fidx)] = markers_init[0][midx(fidx)]
                    marks[1][midx(fidx)] = markers_init[1][midx(fidx)]
            X.val.append(list(alp)+list(ell)+[eps])
            Marks.val.append(marks)
            Stress.val.append(objective)
            gait.append_pose(pf.GeckoBotPose(x, marks, f, cost=objective))
            print(gait.poses[-1].x[0])
        last_obj.val = objective
        return objective

    def constraint1(x):
        """ feet should be at the right position """
        mx, my = model._calc_coords(x, markers_init, f)
        xf, yf = model.get_feet_pos((mx, my))
        xflast, yflast = model.get_feet_pos(markers_init)
        constraint = 0
        for idx in range(n_foot):
            if f[idx]:
                constraint = constraint + \
                    np.sqrt((xflast[idx] - xf[idx])**2 +
                            (yflast[idx] - yf[idx])**2)
        return constraint

    bleg = (blow*len_leg, bup*len_leg)
    btor = (blow*len_tor, bup*len_tor)
    bang = [(alpref[i]-dev_ang, alpref[i]+dev_ang) for i in range(n_limbs)]
    bnds = (bang[0], bang[1], bang[2], bang[3], bang[4],
            bleg, bleg, btor, bleg, bleg,
            (-360, 360))
    con1 = {'type': 'eq', 'fun': constraint1}
    cons = ([con1])
    solution = minimize(objective, x0, method='SLSQP',
                        bounds=bnds, constraints=cons)
    x = solution.x
    mx, my = model._calc_coords(x, markers_init, f)

    constraint = constraint1(x)
    cost = objective(x)
    
    return (x, (mx, my), f, constraint, cost, gait)

 

def animate_gait(fig1, data_xy, data_markers, data_stress, inv=100,
                 col=['red', 'orange', 'green', 'blue', 'magenta', 'darkred']):

    
    max_stress = max(data_stress)
    def update_line(num, data_xy, line_xy, data_markers,
                    lm0, lm1, lm2, lm3, lm4, lm5, leps, data_stress, l_stress):
        x, y = data_xy[num]
        line_xy.set_data(np.array([[x], [y]]))

        xm0, ym0 = data_markers[num][0][0], data_markers[num][1][0]
        xm1, ym1 = data_markers[num][0][1], data_markers[num][1][1]
        xm2, ym2 = data_markers[num][0][2], data_markers[num][1][2]
        xm3, ym3 = data_markers[num][0][3], data_markers[num][1][3]
        xm4, ym4 = data_markers[num][0][4], data_markers[num][1][4]
        xm5, ym5 = data_markers[num][0][5], data_markers[num][1][5]
        lm0.set_data(np.array([[xm0], [ym0]]))
        lm1.set_data(np.array([[xm1], [ym1]]))
        lm2.set_data(np.array([[xm2], [ym2]]))
        lm3.set_data(np.array([[xm3], [ym3]]))
        lm4.set_data(np.array([[xm4], [ym4]]))
        lm5.set_data(np.array([[xm5], [ym5]]))
        leps.set_data(np.array([[xm1, xm4], [ym1, ym4]]))

        
        l_stress.set_color((data_stress[num]/max_stress*1, 0,0))
        l_stress.set_markersize(data_stress[num]/max_stress*50)
        
        return line_xy, lm0, lm1, lm2, lm3, lm4, lm5, leps, l_stress

    n = len(data_xy)
    l_xy, = plt.plot([], [], 'k.', markersize=3)
    msize = 5
    lm0, = plt.plot([], [], 'o', color=col[0], markersize=msize)
    lm1, = plt.plot([], [], 'o', color=col[1], markersize=msize)
    lm2, = plt.plot([], [], 'o', color=col[2], markersize=msize)
    lm3, = plt.plot([], [], 'o', color=col[3], markersize=msize)
    lm4, = plt.plot([], [], 'o', color=col[4], markersize=msize)
    lm5, = plt.plot([], [], 'o', color=col[5], markersize=msize)
    leps, = plt.plot([], [], '-', color='mediumpurple', linewidth=1)
    l_stress, = plt.plot([-10], [0], 'o', color='red', markersize=10)

    minx, maxx, miny, maxy = 0, 0, 0, 0
    for dataset in data_markers:
        x, y = dataset
        minx = min(x) if minx > min(x) else minx
        maxx = max(x) if maxx < max(x) else maxx
        miny = min(y) if miny > min(y) else miny
        maxy = max(y) if maxy < max(y) else maxy
    plt.xlim(minx-5, maxx+5)
    plt.ylim(miny-5, maxy+5)

    line_ani = animation.FuncAnimation(
        fig1, update_line, n, fargs=(data_xy, l_xy, data_markers,
                                     lm0, lm1, lm2, lm3, lm4, lm5, leps,
                                     data_stress, l_stress),
        interval=inv, blit=True)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    return line_ani



x, marks, f, constraint, cost, gait_ = predict_next_pose(
        ref1, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)

x, marks, f, constraint, cost, gait_ = predict_next_pose(
        ref2, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)

x, marks, f, constraint, cost, gait_ = predict_next_pose(
        ref1, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)



gait_alt = pf.GeckoBotGait()
gait_alt.append_pose(initial_pose)
for (xx, marks, stress) in zip(X.val, Marks.val, Stress.val):
    gait_alt.append_pose(pf.GeckoBotPose(xx, marks, ref1[1], cost=stress))


print(gait_.poses[1].x[0])

print('Animate')


fig1 = plt.figure('GeckoBotGaitAnimation')
data_xy, data_markers, data_stress = [], [], []
for pose in gait_alt.poses:
    print(pose.x[0])
    (x, y), (fpx, fpy), (nfpx, nfpy) = \
        pf.get_point_repr(pose.x, pose.markers, pose.f)
    data_xy.append((x, y))
    data_markers.append(pose.markers)
    data_stress.append(pose.cost)
line_ani = animate_gait(fig1, data_xy, data_markers, data_stress)
plt.show('GeckoBotGaitAnimation')


pf.save_animation(line_ani)


gait_alt.plot_gait()
initial_pose.plot()
