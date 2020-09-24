#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:31:41 2020

@author: ls
"""


import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np


from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model


def cut(x):
    return x if x > 0.001 else 0.001


def alpha(x1, x2=0, c1=1):
    alpha = [cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
             cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1),
             x1 + x2*abs(x1),
             cut(45 - x1/2. - abs(x1)*x2/2. + x1*x2*c1),
             cut(45 + x1/2. + abs(x1)*x2/2. + x1*x2*c1)
             ]
    return alpha


# f_l, f_o, f_a = 89, 10, 5.9
f_l, f_o, f_a = [.1, 1, 10]


weight = [f_l, f_o, f_a]
len_leg, len_tor = [9.1, 10.3]

eps = 90
ell = [len_leg, len_leg, len_tor, len_leg, len_leg]
p1 = (0, 0)

f2 = [0, 1, 1, 0]
f1 = [1, 0, 0, 1]

incr = 20
ref = []


Q1 = 89
Q2 = .4

for q1 in np.linspace(-Q1, Q1, incr):
    ref.append([alpha(q1, Q2), f2])
ref.append([alpha(q1, Q2), [1, 1, 1, 1]])

idx1 = len(ref)

for q1 in np.linspace(Q1, -Q1, incr):
    ref.append([alpha(q1, Q2), f1])
ref.append([alpha(q1, Q2), [1, 1, 1, 1]])

idx2 = len(ref)

for q1 in np.linspace(-Q1, Q1, incr):
    ref.append([alpha(q1, Q2), f2])
ref.append([alpha(q1, Q2), [1, 1, 1, 1]])

idx3 = len(ref)

for q1 in np.linspace(Q1, -Q1, incr):
    ref.append([alpha(q1, Q2), f1])
ref.append([alpha(q1, Q2), [1, 1, 1, 1]])

idx4 = len(ref)

ref.append([alpha(q1, Q2), f2])



init_pose = pf.GeckoBotPose(*model.set_initial_pose(
                            ref[0][0], eps, p1,
                            len_leg=len_leg, len_tor=len_tor))
gait = pf.predict_gait(ref, init_pose, weight, (len_leg, len_tor))


gait.plot_gait()

# %%

header_0 = """
\\documentclass[tikz]{standalone}
\\begin{document}
"""




ending = """
\\end{tikzpicture}
"""

ending_0 = """
\\end{document}
"""

ani_str = header_0

xpos1, ypos1 = gait.poses[idx1].get_m1_pos()
xpos2, ypos2 = gait.poses[idx2].get_m1_pos()
xpos3, ypos3 = gait.poses[idx3].get_m1_pos()
xpos4, ypos4 = gait.poses[idx4].get_m1_pos()

colors = pf.get_actuator_tikzcolor()
for pose in gait.poses:
    xpos, ypos = pose.get_m1_pos()
    eps_ = -(pose.get_eps() - eps)
    header = """
\\begin{tikzpicture}[scale=.1, rotate=%s]
\\path[use as bounding box](-15,-25) rectangle (20,10);
""" % eps_
    init = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos, 4), round(-ypos, 4))
    init1 = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos1, 4), round(-ypos+ypos1, 4))
    init1_ = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos1, 4), round(-ypos-ypos1, 4))
    init2 = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos2, 4), round(-ypos+ypos2, 4))
    init2_ = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos2, 4), round(-ypos-ypos2, 4))
    init3 = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos3, 4), round(-ypos+ypos3, 4))
    init3_ = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos3, 4), round(-ypos-ypos3, 4))
    init4 = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos4, 4), round(-ypos+ypos4, 4))
    init4_ = '\\path[fill=gray!50] (%f,%f)circle(1);\n' % (round(-xpos+xpos4, 4), round(-ypos-ypos4, 4))
    ani_str += (header + init + init1 + init2 + init3 + init4 
                + init1_ + init2_ + init3_ + init4_
                + pose.get_tikz_repr(R=.7, col=colors, yshift=-ypos,
                                     xshift=-xpos,
                                     rotate=None,
#                                     rotate=eps_
                                     ) + ending)

filename = 'Out/curved_gait_shift_%f_%f.tex' % (Q1, Q2)

with open(filename, 'w') as fout:
    fout.writelines(ani_str + ending_0)
