# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:45:11 2019

@author: AmP
"""

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    from os import path
    sys.path.insert(0, path.dirname(path.dirname(
            path.abspath(__file__))))

    from Src.TrajectoryPlanner import optimal_planner as opt
    from Src.TrajectoryPlanner import rotate_on_spot as rotspot
    from Src.Utils import plot_fun as pf
    from Src.Utils import save
    from Src.Math import kinematic_model as model


# %%

    alpha = [90, 0, -90, 90, 0]
    feet = [1, 0, 0, 1]
    eps = 90
    p1 = (0, 0)
    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    pattern = [
        ([90, 0, -90, 90, 0], [1, 0, 0, 1]),
        ([45, 45, 0, 45, 45], [1, 0, 0, 1]),
        ([45, 45, 0, 45, 45], [1, 1, 1, 1]),
        ([45, 45, 0, 45, 45], [1, 1, 0, 0]),
        ([45, 45, 90, 45, 45], [1, 1, 0, 0]),
        ([45, 45, 90, 45, 45], [1, 1, 1, 1]),
        ([45, 45, 90, 45, 45], [0, 0, 1, 1]),
        ([45, 45, -90, 45, 45], [0, 0, 1, 1]),
        ([45, 45, -90, 45, 45], [1, 1, 1, 1]),
        ([45, 45, -90, 45, 45], [1, 1, 0, 0]),
        ([45, 45, 0, 45, 45], [1, 1, 0, 0]),
        ([45, 45, 0, 45, 45], [1, 1, 1, 1]),
        ([45, 45, 0, 45, 45], [1, 0, 0, 1]),
        ([90, 0, -90, 90, 0], [1, 0, 0, 1]),
            ]

    pattern2 = [
        ([90, 0, -90, 90, 0], [1, 0, 0, 1]),
        ([45, 45, -90, 45, 45], [1, 0, 0, 1]),
        ([45, 45, -90, 45, 45], [1, 1, 1, 1]),
        ([45, 45, -90, 45, 45], [0, 0, 1, 1]),
        ([45, 45, 90, 45, 45], [0, 0, 1, 1]),
        ([45, 45, 90, 45, 45], [1, 1, 1, 1]),
        ([45, 45, 90, 45, 45], [0, 0, 1, 1]),
        ([45, 45, -90, 45, 45], [1, 1, 0, 0]),
        ([45, 45, -90, 45, 45], [1, 1, 1, 1]),
        ([45, 45, -90, 45, 45], [1, 0, 0, 1]),
        ([90, 0, -90, 90, 0], [1, 0, 0, 1]),
            ]



    for pose in pattern2:
        alpha, feet = pose

        act_pose = gait.poses[-1]
        x, y = act_pose.markers
        act_pos = (x[1], y[1])

        predicted_pose = model.predict_next_pose(
                    [alpha, feet], act_pose.x, (x, y))
        predicted_pose = pf.GeckoBotPose(*predicted_pose)
        gait.append_pose(predicted_pose)


    ## %% Plots
    gait.plot_gait()
#    gait.plot_markers(1)
#        gait.plot_com()
#        plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)
    plt.axis('off')

    gait_str = gait.get_tikz_repr(shift=2.9)
    save.save_plt_as_tikz('Scripts/principle_sketch.tex', gait_str)


# %% Test algorithm

    for idx, init in enumerate([[[90, 0, -90, 90, 0], [1, 0, 0, 1]],
                           [[0, 90, 90, 0., 90], [0, 1, 1, 0]]]):
        for jdx, xref in enumerate([(-2, -3), (2, -3)]):

            alpha, feet = init

            eps = 90
            p1 = (0, 0)
            x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
            initial_pose = pf.GeckoBotPose(x, (mx, my), f)
            gait = pf.GeckoBotGait()
            gait.append_pose(initial_pose)

            xbar = opt.xbar(xref, p1, eps)
            pattern = rotspot.rotate_on_spot(xbar, alpha, feet)

            for pose in pattern:
                alpha, feet, ptime = pose
                act_pose = gait.poses[-1]
                x, y = act_pose.markers
                act_pos = (x[1], y[1])

                predicted_pose = model.predict_next_pose(
                            [alpha, feet], act_pose.x, (x, y))
                predicted_pose = pf.GeckoBotPose(*predicted_pose)
                gait.append_pose(predicted_pose)

        #    gait.plot_gait()
            plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)
            plt.axis('off')

            gait_str = gait.get_tikz_repr(shift=2.9)
            save.save_plt_as_tikz(
                    'Scripts/test_rot_on_spot_{}_{}.tex'.format(idx, jdx),
                    gait_str)

    # %% 4 Cases

    # %% 0

    alpha = [90, 0, -90, 90, 0]
    feet = [1, 0, 0, 1]
    xref = (2, -1)
    eps = 90
    p1 = (0, 0)

    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), feet)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

#    gait.plot_gait()
    plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)

    plt.axis('off')
    gait_str = gait.get_tikz_repr()
    save.save_plt_as_tikz('Scripts/case_0.tex', gait_str)

    # %% 1

    alpha = [90, 0, -90, 90, 0]
    feet = [1, 0, 0, 1]
    xref = (-2, -1)
    eps = 90
    p1 = (0, 0)

    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), feet)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

#    gait.plot_gait()
    plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)

    plt.axis('off')
    gait_str = gait.get_tikz_repr()
    save.save_plt_as_tikz('Scripts/case_1.tex', gait_str)

    # %% 2

    alpha = [0, 90, 90, 0, 90]
    feet = [0, 1, 1, 0]
    xref = (2, -1)
    eps = 90
    p1 = (0, 0)

    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), feet)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

#    gait.plot_gait()
    plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)

    plt.axis('off')
    gait_str = gait.get_tikz_repr()
    save.save_plt_as_tikz('Scripts/case_2.tex', gait_str)

    # %% 3

    alpha = [0, 90, 90, 0, 90]
    feet = [0, 1, 1, 0]
    xref = (-2, -1)
    eps = 90
    p1 = (0, 0)

    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), feet)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

#    gait.plot_gait()
    plt.plot(xref[0], xref[1], marker='o', color='red', markersize=12)

    plt.axis('off')
    gait_str = gait.get_tikz_repr()
    save.save_plt_as_tikz('Scripts/case_3.tex', gait_str)