# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:34:00 2020

@author: AmP


convert -density 500 -delay 8 -loop 0 -alpha remove in.pdf out.gif

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

# %%
# f_l, f_o, f_a = 89, 10, 5.9
f_l, f_o, f_a = [.1, 1, 10]


weight = [f_l, f_o, f_a]
len_leg, len_tor = [9.1, 10.3]

eps = 90
ell = [len_leg, len_leg, len_tor, len_leg, len_leg]
p1 = (0, 0)

f2 = [0, 1, 1, 0]
f1 = [1, 0, 0, 1]
fa = [1, 1, 1, 1]

f4f = [1, 1, 1, 0]
f1f = [0, 1, 1, 1]
f2f = [1, 0, 1, 1]
f3f = [1, 1, 0, 1]


incr = 20
ref = []

# first movement
for q1 in np.linspace(-89, 89, incr):
    ref.append([alpha(q1), f2])
ref.append([alpha(q1), [1, 1, 1, 1]])

# special treat for rear right:
for aa in np.linspace(89.5, .5, 3):
    alp_ = alpha(q1)
    alp_[4] = aa
    ref.append([alp_, f4f])
for aa in np.linspace(.5, 89.5, 3):
    alp_ = alpha(q1)
    alp_[4] = aa
    ref.append([alp_, f4f])


# special treat for front left:
for aa in np.linspace(.5, 89.5, 3):
    alp_ = alpha(q1)
    alp_[0] = aa
    ref.append([alp_, f1f])
for aa in np.linspace(89.5, .5, 3):
    alp_ = alpha(q1)
    alp_[0] = aa
    ref.append([alp_, f1f])
ref.append([alpha(q1), [1, 1, 1, 1]])

#
# second movement
for q1 in np.linspace(89, -89, incr):
    ref.append([alpha(q1), f1])
ref.append([alpha(q1), [1, 1, 1, 1]])


# special treat for rear left:
for aa in np.linspace(89, 1, 3):
    alp_ = alpha(q1)
    alp_[3] = aa
    ref.append([alp_, f3f])
for aa in np.linspace(.5, 89.5, 3):
    alp_ = alpha(q1)
    alp_[3] = aa
    ref.append([alp_, f3f])

# special treat for front right:
for aa in np.linspace(.5, 89.5, 3):
    alp_ = alpha(q1)
    alp_[1] = aa
    ref.append([alp_, f2f])
for aa in np.linspace(89.5, .5, 3):
    alp_ = alpha(q1)
    alp_[1] = aa
    ref.append([alp_, f2f])

ref.append([alpha(q1), [1, 1, 1, 1]])
ref.append([alpha(q1), f2])

# %%
# simulate
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

filename = '../../Out/Animations/climbing_gait_shift.tex'

with open(filename, 'w') as fout:
    fout.writelines(ani_str + ending_0)


