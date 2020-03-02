# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:48:39 2020

@author: AmP
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


def alpha(x1, f, x2=0, c1=1):
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
for q1 in np.linspace(-89, 89, incr):
    ref.append([alpha(q1, f2), f2])
ref.append([alpha(q1, f2), [1, 1, 1, 1]])

for q1 in np.linspace(89, -89, incr):
    ref.append([alpha(q1, f1), f1])
ref.append([alpha(q1, f1), [1, 1, 1, 1]])
ref.append([alpha(q1, f1), f2])

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

header = """
\\begin{tikzpicture}[scale=.1]
\\path[use as bounding box](-11,-18) rectangle (9.5,10);
"""


ending = """
\\end{tikzpicture}
"""

ending_0 = """
\\end{document}
"""

ani_str = header_0

colors = pf.get_actuator_tikzcolor()
for pose in gait.poses:
    _, ypos = pose.get_m1_pos()
    init = '\\path[fill=gray!50] (0,%f)circle(1);\n' % round(-ypos, 4)
    init2 = '\\path[fill=gray!50] (0,12.703 + %f)circle(1);\n' % round(-ypos, 4)
    init3 = '\\path[fill=gray!50] (0,-12.703 + %f)circle(1);\n' % round(-ypos, 4)
    ani_str += (header + init + init2 + init3
                + pose.get_tikz_repr(R=.7, col=colors, yshift=-ypos) + ending)

filename = '../../Out/Animations/straight_gait_shift.tex'

with open(filename, 'w') as fout:
    fout.writelines(ani_str + ending_0)
