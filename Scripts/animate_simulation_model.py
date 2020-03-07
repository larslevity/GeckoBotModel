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



eps = 90
ell = [9.3, 9.3, 11.2, 9.3, 9.3]
p1 = (0, 0)

# REF straight
alp = [5, 85, 87.5, 5, 85]
ref1 = [[85, 5, -87.5, 85, 5], [1, 0, 0, 1]]
ref2 = [[5, 85, 87.5, 5, 85], [0, 1, 1, 0]]


# REF curve
alp = [5, 5, -24, 5, 5]
ref1 = [[164, 124, -152, 221, 62], [1, 0, 0, 1]]
ref2 = [[5, 5, -24, 5, 5], [0, 1, 1, 0]]



x, marks, f = model.set_initial_pose(
        alp, eps, p1, len_leg=ell[0], len_tor=ell[2])

ref1_pose = pf.GeckoBotPose(*model.set_initial_pose(
        ref1[0], eps, (15, -8), len_leg=ell[0]/5, len_tor=ell[2]/5))
ref2_pose = pf.GeckoBotPose(*model.set_initial_pose(
        ref2[0], eps, (15, -8), len_leg=ell[0]/5, len_tor=ell[2]/5))


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



gait_alt = pf.GeckoBotGait()
gait_alt.append_pose(initial_pose)

x, marks, f1, constraint, cost, gait_ = predict_next_pose(
        ref1, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)
idx1 = len(X.val)

x, marks, f2, constraint, cost, gait_ = predict_next_pose(
        ref2, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)
idx2 = len(X.val)

x, marks, f1, constraint, cost, gait_ = predict_next_pose(
        ref1, x, marks, len_leg=9.3, len_tor=11.2, gait=gait)
idx3 = len(X.val)


for idx, (xx, marks, stress) in enumerate(zip(X.val, Marks.val, Stress.val)):
    f = f1
    if idx > idx1:
        f = f2
        if idx > idx2:
            f = f1
    gait_alt.append_pose(pf.GeckoBotPose(xx, marks, f, cost=stress))


print(gait_.poses[1].x[0])
#%%
# Meta Data
data_xy, data_markers, data_stress, lims = [], [], [], [-1, 18, -11, 5]
for pose in gait_alt.poses:
    # print(pose.x[0])
    (x, y), (fpx, fpy), (nfpx, nfpy) = \
        pf.get_point_repr(pose.x, pose.markers, pose.f)
    data_xy.append((x, y))
    data_markers.append(pose.markers)
    data_stress.append(pose.cost)
    # check lims
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    if lims[0] > xmin:
        lims[0] = xmin
    if lims[1] < xmax:
        lims[1] = xmax
    if lims[2] > ymin:
        lims[2] = ymin
    if lims[3] < ymax:
        lims[3] = ymax


# %% TikZ Pic 

header_0 = """
\\documentclass[tikz]{standalone}
\\begin{document}
"""
mar = 2
header = """
\\begin{tikzpicture}[scale=.1]
\\path[use as bounding box](%f,%f) rectangle (%f,%f);
""" % (lims[0]-mar, lims[2]-mar, lims[1]+mar, lims[3]+mar)


ending = """
\\end{tikzpicture}
"""

ending_0 = """
\\end{document}
"""

init_str = initial_pose.get_tikz_repr(R=.7, col='gray!50')
init2_str = gait_alt.poses[idx1].get_tikz_repr(R=.7, col='gray!50')
init3_str = gait_alt.poses[idx2].get_tikz_repr(R=.7, col='gray!50')
ref1_str = ref1_pose.get_tikz_repr(R=.35, col='blue!50', linewidth=.35, dashed=0)
ref2_str = ref2_pose.get_tikz_repr(R=.35, col='blue!50', linewidth=.35, dashed=0)

ani_str = header_0

colors = pf.get_actuator_tikzcolor()
max_stress = max(data_stress)

for idx, pose in enumerate(gait_alt.poses):
    ref = ref1_str
    init = init_str
    if idx > idx1:
        ref = ref2_str
        init = init2_str
        if idx > idx2:
            ref = ref1_str
            init = init3_str
    _, ypos = pose.get_m1_pos()
    stress = '\\path[fill=red!50] (15,5)circle(%s)node[red, scale=.75]{stress};\n' % round(pose.cost/max_stress*5, 4)
    ref_note = '\\path (15, -3)node[blue, scale=0.75]{ref};\n';
    ani_str += (header + stress + init + ref + ref_note
                + pose.get_tikz_repr(R=.7, col=colors) + ending)

filename = '../Out/Animations/simulation_model_animation_curve.tex'

with open(filename, 'w') as fout:
    fout.writelines(ani_str + ending_0)

# To convert:
# convert -density 500 -delay 8 -loop 0 -alpha remove in.pdf out.gif






# %%
#print('Animate')
#
#
#fig1 = plt.figure('GeckoBotGaitAnimation')
#line_ani = animate_gait(fig1, data_xy, data_markers, data_stress)
#plt.show('GeckoBotGaitAnimation')
#
#
#pf.save_animation(line_ani)
#
#
#gait_alt.plot_gait()
#initial_pose.plot()

