# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:42:20 2019

@author: AmP
"""

from Src.TrajectoryPlanner import optimal_planner as op
from Src.Math import kinematic_model as model
from Src.Utils import plot_fun as pf

alp = [90, 0, -90, 90, 0]
eps = 0
ell = [1, 1, 1.2, 1, 1]
p1 = (0, 0)

x, marks, feet = model.set_initial_pose(
        alp, eps, p1, len_leg=ell[0], len_tor=ell[2])
feet = [1, 0, 0, 1]

initial_pose = pf.GeckoBotPose(x, marks, feet)
gait = pf.GeckoBotGait()
gait.append_pose(initial_pose)

p_ref = (10, 0)

alp_ref, feet_ref = op.optimal_planner(p_ref, alp, feet, (marks, eps),
                                       f=[100, 10, 10])

x, marks, f, stats = model.predict_next_pose(
    [alp_ref, feet_ref], x, marks, len_leg=1, len_tor=1.2)
gait.append_pose(
    pf.GeckoBotPose(x, marks, f, constraint=stats[0], cost=stats[1]))

gait.plot_gait()
