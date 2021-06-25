#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:54:46 2021

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
from Src.TrajectoryPlanner.optimal_planner import alpha


f_l, f_o, f_a = [.1, 1, 10]


weight = [f_l, f_o, f_a]
len_leg, len_tor = [9.1, 10.3]

eps = 90
ell = [len_leg, len_leg, len_tor, len_leg, len_leg]

roboscale=.125
p1 = (6/roboscale,  0)
p2 = (-6/roboscale, 0)




# %%

header_0 = """
\\documentclass[tikz]{standalone}
%% This file was create by 'GeckoBotModel/Scripts/Animation_gaitlaw/animation_gaitlaw.py'
\\usepackage[sfdefault, light]{FiraSans}
\\usepackage{bm}
\\begin{document}
"""

header = """
\\begin{tikzpicture}[scale=1]
\\path[use as bounding box](-8,-4.5) rectangle ++(16,9);
"""

colors = pf.get_actuator_tikzcolor()

def pose(q1, q2):
    pose1 = pf.GeckoBotPose(*model.set_initial_pose(
                            alpha(q1, q2), eps, p1,
                            len_leg=len_leg, len_tor=len_tor)).get_tikz_repr(R=.7, col=colors)
    pose2 = pf.GeckoBotPose(*model.set_initial_pose(
                            alpha(-q1, q2), eps, p2,
                            len_leg=len_leg, len_tor=len_tor)).get_tikz_repr(R=.7, col=colors)
    label1 = "\n\\path (OM)++(0,15)node{$\\bm{r}(q_1, q_2)$};"
    label2 = "\n\\path (OM)++(0,15)node{$\\bm{r}(-q_1, q_2)$};"
    return "\n\\begin{scope}[scale=%s]"%roboscale + pose1 + label1 + pose2 +label2 + "\n\\end{scope}"



def axis(q1, q2, c1=1):
    q1, q2 = round(q1, 1), round(q2, 2)
    axis = """
\\begin{scope}[xscale=1/90*4, yscale=1/90*2.5]
\\draw[-latex] (-100, 0)--(100, 0)node[right]{$q_1$};
\\draw[-latex] (0, -100)--(0, 100)node[above]{$\\bar{\\bm{\\alpha}}$};

\\draw[red]             (-100, 45 - -100/2 - 100*%s /2 + -100*%s ) --(0,45)node[pos=.25, sloped, above]{\\footnotesize $\\bar{\\alpha}_0$}-- (100, 45 - 100/2 - 100*%s /2 + 100*%s ) ;
\\draw[red!50!black]    (-100, 45 + -100/2 + 100*%s /2 + -100*%s ) --(0,45)node[pos=.25, sloped, above]{\\footnotesize $\\bar{\\alpha}_1$}-- (100, 45 + 100/2 + 100*%s /2 + 100*%s ) ;
\\draw[orange]          (-100, -100 + 100*%s )                     --(0,0)node[pos=.3, sloped, above]{\\footnotesize $\\bar{\\alpha}_2$}-- (100, 100 + 100*%s ) ;
\\draw[blue]            (-100, 45 - -100/2 - 100*%s /2 + -100*%s ) --(0,45)node[pos=.5, sloped, above]{\\footnotesize $\\bar{\\alpha}_3$}-- (100, 45 - 100/2 - 100*%s /2 + 100*%s ) ;
\\draw[blue!50!black]   (-100, 45 + -100/2 + 100*%s /2 + -100*%s ) --(0,45)node[pos=.5, sloped, above]{\\footnotesize $\\bar{\\alpha}_4$}-- (100, 45 + 100/2 + 100*%s /2 + 100*%s ) ;

\\fill[red]             (%s, %s)node{$\\bullet$};
\\fill[red!50!black]    (%s, %s)node{$\\bullet$};
\\fill[orange]          (%s, %s)node{$\\bullet$};
\\fill[blue]            (%s, %s)node{$\\bullet$};
\\fill[blue!50!black]   (%s, %s)node{$\\bullet$};

\\fill[red]             (%s, %s)node{$\\bullet$};
\\fill[red!50!black]    (%s, %s)node{$\\bullet$};
\\fill[orange]          (%s, %s)node{$\\bullet$};
\\fill[blue]            (%s, %s)node{$\\bullet$};
\\fill[blue!50!black]   (%s, %s)node{$\\bullet$};

\\path (-15, 140)node[right,align=left, draw=gray!50]{$q_2 = %s$};
\\end{scope}

""" % (q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,q2,
       q1,45 - q1/2. - abs(q1)*q2/2. + q1*q2*c1,
       q1,45 + q1/2. + abs(q1)*q2/2. + q1*q2*c1,
       q1,q1 + q2*abs(q1),
       q1,45 - q1/2. - abs(q1)*q2/2. + q1*q2*c1,
       q1,45 + q1/2. + abs(q1)*q2/2. + q1*q2*c1,
#       
       -q1,45 - -q1/2. - abs(q1)*q2/2. + -q1*q2*c1,
       -q1,45 + -q1/2. + abs(q1)*q2/2. + -q1*q2*c1,
       -q1,-q1 + q2*abs(q1),
       -q1,45 - -q1/2. - abs(q1)*q2/2. + -q1*q2*c1,
       -q1,45 + -q1/2. + abs(q1)*q2/2. + -q1*q2*c1,
#       
       q2
       )
    return axis


ending = """
\\end{tikzpicture}
"""

ending_0 = """
\\end{document}
"""

ani_str = header_0


q1 = 85
for q2 in np.concatenate((np.linspace(0, .5, 51), np.linspace(.49, -.5, 100), 
#                          np.linspace(-.4, -.1, 4)
                          )):
    ani_str += (header + axis(q1, q2) + pose(q1, q2) + ending)

#for pose in gait.poses:
#    _, ypos = pose.get_m1_pos()
#    init = '\\path[fill=gray!50] (0,%f)circle(1);\n' % round(-ypos, 4)
#    init2 = '\\path[fill=gray!50] (0,12.703 + %f)circle(1);\n' % round(-ypos, 4)
#    init3 = '\\path[fill=gray!50] (0,-12.703 + %f)circle(1);\n' % round(-ypos, 4)
#    ani_str += (header + axis(pose.alp[2]) + init + init2 + init3
#                + pose.get_tikz_repr(R=.7, col=colors, yshift=-ypos) + ending)

filename = '../../Out/Animations/gaitlaw.tex'

with open(filename, 'w') as fout:
    fout.writelines(ani_str + ending_0)
