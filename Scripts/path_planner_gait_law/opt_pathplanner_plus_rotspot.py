# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:31:42 2019

@author: AmP
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    from os import path
    sys.path.insert(0, path.dirname(path.dirname(path.dirname(
            path.abspath(__file__)))))

    from Src.TrajectoryPlanner import optimal_planner as opt
    from Src.Utils import plot_fun as pf
    from Src.Utils import save
    from Src.Math import kinematic_model as model
    from Src.TrajectoryPlanner import rotate_on_spot as rotspot

    for replay in range(10):

#        alpha = [90, 0, -90, 90, 0]
#        feet = [1, 0, 0, 1]
        alpha = [0, 90, 90, 0, 90]
        feet = [0, 1, 1, 0]
        eps = 90
        p1 = (0, 0)
        x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
        initial_pose = pf.GeckoBotPose(x, (mx, my), f)
        gait = pf.GeckoBotGait()
        gait.append_pose(initial_pose)

        xref = (10, -3)

        n = 1

        def calc_dist(pose, xref):
            mx, my = pose.markers
            act_pos = np.r_[mx[1], my[1]]
            dpos = xref - act_pos
            return np.linalg.norm(dpos)

        def add_noise(alpha):
            return list(np.r_[alpha]+np.random.normal(0, 5, 5))

        i = 0
        while calc_dist(gait.poses[-1], xref) > .7:
            act_pose = gait.poses[-1]
            x, y = act_pose.markers
            act_pos = (x[1], y[1])
            eps = act_pose.x[-1]
            alp_act = act_pose.alp
            xbar = opt.xbar(xref, act_pos, eps)
            deps = np.rad2deg(np.arctan2(xbar[1], xbar[0]))
            if abs(deps) > 70:
                pattern = rotspot.rotate_on_spot(xbar, alp_act, feet)
                for pose in pattern:
                    alpha, feet, ptime = pose
                    alpha = add_noise(alpha)
                    act_pose = gait.poses[-1]
                    x, y = act_pose.markers
                    act_pos = (x[1], y[1])
                    predicted_pose = model.predict_next_pose(
                                [alpha, feet], act_pose.x, (x, y))
                    predicted_pose = pf.GeckoBotPose(*predicted_pose)
                    gait.append_pose(predicted_pose)
            else:
                alpha, feet = opt.optimal_planner(xbar, alp_act, feet, n,
                                                  show_stats=1)
                alpha = add_noise(alpha)
                predicted_pose = model.predict_next_pose(
                        [alpha, feet], act_pose.x, (x, y))

                predicted_pose = pf.GeckoBotPose(*predicted_pose)
                gait.append_pose(predicted_pose)
            i += 1
            print('pose', i, 'dist: \t', round(calc_dist(gait.poses[-1],
                                                         xref), 2), '\n')
            if i > 50:
                break

    # %% Plots
#    gait.plot_gait()
        gait.plot_markers(1)
#        gait.plot_com()
    plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)
    plt.axis('off')

    gait_str = gait.get_tikz_repr()
    save.save_plt_as_tikz('Scripts/gait_{}x.tex'.format(replay),
                          gait_str)

    # %% Animation

    #    gait.animate()
