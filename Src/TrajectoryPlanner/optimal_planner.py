# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:51:44 2019

@author: AmP
"""

import numpy as np
from scipy.optimize import minimize

from Src.Math import kinematic_model as model
from Src.Math import util_funs as uf


def optimal_planner(p_ref, alp_act, feet_act, (marks_act, eps_act),
                    dist_min=1, len_leg=1, len_tor=1.2,
                    f=[100, .1, 10]):
    """
    calculates optimal reference to bring the robot near to *p_ref*,
    where *p_ref* describes the reference position of the robot in
    task space, i.e. cartesian coordinates.

    The variable to be optimized therefor is:
        x = [alp0, alp1, alp2, alp3, alp4, alp5]
    """
    weight_dist = .5  # factor for weighting dist against orientation
    ell = [len_leg, len_leg, len_tor, len_leg, len_leg]
    alp_act = model._check_alpha(alp_act)
    x_init = model.flat_list([alp_act, ell, [eps_act]])

    p_ref = np.r_[p_ref]
    pos_act = np.r_[marks_act[1][0], marks_act[1][1]]
    dpos_act = p_ref - pos_act
    dist_act = np.linalg.norm(dpos_act)

    dir_act = np.r_[np.cos(np.radians(eps_act)),
                    np.sin(np.radians(eps_act))]
    deps_act = uf.calc_angle(dpos_act, dir_act)

    # Are we already there?
    if dist_act < dist_min:
        alp_ref = alp_act
        feet_ref = feet_act
        return [alp_ref, feet_ref]

    # Not the case -> lets optimize
    x0 = alp_act

    feet_ref = [not(foot) for foot in feet_act]

    def objective(x):
        print([round(xx, 2) for xx in x])

        x_opt, marks_opt, _, _ = model.predict_next_pose(
                [x, feet_ref], x_init, marks_act, f=f,
                len_leg=len_leg, len_tor=len_tor)
        eps_opt = x_opt[-1]

        pos_opt = np.r_[marks_opt[1][0], marks_opt[1][1]]
        dpos_opt = p_ref - pos_opt
        dist_opt = np.linalg.norm(dpos_opt)

        dir_opt = np.r_[np.cos(np.radians(eps_opt)),
                        np.sin(np.radians(eps_opt))]
        deps_opt = uf.calc_angle(dpos_opt, dir_opt)

        cost = (weight_dist*(dist_opt/dist_act))
#                + (1 - weight_dist)*abs(deps_opt/180.))
        print(round(cost, 3))
        return cost

    bnds = [(0, 120)]*5

    def jac(x):
        return np.r_[x[2], x[2], abs(x[2]), x[2], x[2]]

    solution = minimize(objective, x0, method='Newton-CG',
                        jac=jac)
    return solution.x, feet_ref
