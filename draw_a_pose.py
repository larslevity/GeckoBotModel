# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:25:55 2019

@author: AmP
"""

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model


def alpha1(x1, x2, f):
    alpha = [(45 - x1/2. - abs(x1)*x2/2. + (f[0])*x1*x2),
             (45 + x1/2. + abs(x1)*x2/2. + (f[1])*x1*x2),
             x1 + x2*abs(x1),
             (45 - x1/2. - abs(x1)*x2/2. + (f[2])*x1*x2),
             (45 + x1/2. + abs(x1)*x2/2. + (f[3])*x1*x2)
             ]
    return alpha


eps = 90
alp1 = [85, 5, -80, 85, 5]
alp2 = [5, 85,  80, 5, 85]
f1 = [1, 0, 0, 1]
f2 = [0, 1, 1, 0]



pose1 = pf.GeckoBotPose(*model.set_initial_pose(alp1, eps, (0, 0)))
pose1.f = f1
pose1.save_as_tikz('pose1', compileit=False)

pose2 = pf.GeckoBotPose(*model.set_initial_pose(alp2, eps, (0, 0)))
pose2.f = f2
pose2.save_as_tikz('pose2', compileit=False)


x1 = -80
x2 = -.5
alp3 = alpha1(x1, x2, f1)
pose3 = pf.GeckoBotPose(*model.set_initial_pose(alp3, eps, (0, 0)))
pose3.f = f1
pose3.save_as_tikz('pose3', compileit=False)


x1 = 80
alp4 = alpha1(x1, x2, f2)
pose4 = pf.GeckoBotPose(*model.set_initial_pose(alp4, eps, (0, 0)))
pose4.f = f2
pose4.save_as_tikz('pose4', compileit=False)


x1 = -80
x2 = .5
alp5 = alpha1(x1, x2, f1)
pose5 = pf.GeckoBotPose(*model.set_initial_pose(alp5, eps, (0, 0)))
pose5.f = f1
pose5.save_as_tikz('pose5', compileit=False)
alp5a = alpha1(-x1, x2, f2)
pose5a = pf.GeckoBotPose(*model.predict_next_pose(
        [alp5a, f2], pose5.x, pose5.markers))
pose5a.save_as_tikz('pose5a', compileit=False)

x1 = 80
alp6 = alpha1(x1, x2, f2)
pose6 = pf.GeckoBotPose(*model.set_initial_pose(alp6, eps, (0, 0)))
pose6.f = f2
pose6.save_as_tikz('pose6', compileit=False)
