# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:20:57 2020

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
    from Src import calibration

    # MODEL PARAMS
    l_leg, l_tor = calibration.get_len('vS11')

    f_l, f_o, f_a = [89, 10, 5.9]  # exp c110 redo fit

    for replay in range(1):

        alpha = [30, 0, -30, 30, 0]
        lastq1 = -30
        feet = [1, 0, 0, 1]
        eps = 90
        p1 = (0, 0)
        x, (mx, my), f = model.set_initial_pose(alpha, eps, p1,
                                                len_leg=l_leg, len_tor=l_tor)
        initial_pose = pf.GeckoBotPose(x, (mx, my), f)
        gait = pf.GeckoBotGait()
        gait.append_pose(initial_pose)

        xref = (70, 100)

        XREF = [(65, 0), (65, 65), (0, 65), (0, 110)]

        nmax = 4
        Q1, Q2, DIST = [], [], []

        def calc_dist(pose, xref):
            mx, my = pose.markers
            act_pos = np.r_[mx[1], my[1]]
            dpos = xref - act_pos
            return np.linalg.norm(dpos)

        def add_noise(alpha):
            return list(np.r_[alpha]+np.random.normal(0, 5, 5))

        i = 0
        for xref in XREF:
            dist = calc_dist(gait.poses[-1], xref)
            while dist > 10:
                act_pose = gait.poses[-1]
                x, y = act_pose.markers
                act_pos = (x[1], y[1])
                eps = act_pose.x[-1]
                alp_act = act_pose.alp
                xbar = opt.xbar(xref, act_pos, eps)
                deps = np.rad2deg(np.arctan2(xbar[1], xbar[0]))

                [alpha, feet], q = opt.optimal_planner_nhorizon(
                        xbar, alp_act, feet, lastq1,
                        nmax=nmax, show_stats=1, q1bnds=[50, 90])
                lastq1 = q[0]
                Q1.append(abs(q[0]))
                Q2.append(q[1])
    #            alpha = add_noise(alpha)
                predicted_pose = model.predict_next_pose(
                        [alpha, feet], act_pose.x, (x, y),
                        f=[f_l, f_o, f_a],
                        len_leg=l_leg, len_tor=l_tor)

                predicted_pose = pf.GeckoBotPose(*predicted_pose)
                gait.append_pose(predicted_pose)
                i += 1

                dist = calc_dist(gait.poses[-1], xref)
                DIST.append(dist)

                print('pose', i, 'dist: \t', round(dist, 2), '\n')
                if i > 100:
                    break

    # %% Plots
#    gait.plot_gait()
    gait.plot_markers(1)


    plt.plot(p1[0], p1[1], marker='o', color='k', mfc='orange', markersize=10)
    plt.text(p1[0]+2, p1[1]+2, str('start'), fontsize=30)

    for i, (x, y) in enumerate(XREF):
        plt.plot(x, y, marker='o', color='black', markersize=12, mfc='red')
        plt.text(x+2, y+2, 'Goal '+str(i), fontsize=30)
    
    
    plt.grid()
    plt.yticks([0, 65, 110.002], ['0', '65', '110'])
    plt.xticks([0, 65.002], ['0', '65'])
    plt.ylim((-25, 120))
    plt.xlim((-25, 110))
    plt.axis('scaled')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    gait_str = gait.get_tikz_repr()

    gait_str += '\\definecolor{color0}{rgb}{1,0.647058823529412,0}\n'
    gait_str += '\\draw[color0, line width=1mm, -latex] (0,0)--(0,10);'

    kwargs = {'extra_axis_parameters':
              {'x=.1cm', 'y=.1cm', 'anchor=origin', 'xmin=-25',
               'xmax=110','axis line style={draw opacity=0}',
               'ymin=-25, ymax=120', 'tick pos=left',}}
    save.save_plt_as_tikz('Out/opt_pathplanner/gait.tex',
                          additional_tex_code=gait_str, 
                          scope='scale=.1, opacity=1', **kwargs)

    # %%
    plt.figure('Q1Q2')
    plt.plot(Q1, 'b')

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(Q2, 'r')
    ax2.set_ylim(-.7, .7)
    ax2.set_yticks([-.5, 0, .5])
    ax1.set_ylim(40, 100)

    ax1.grid()

    n = 0
    for idx, d in enumerate(DIST):
        if d < 10:
            ax1.plot([idx, idx], [43, 95], 'gray')
            ax1.text(idx-1, 43, 'Reached {}'.format(n),  horizontalalignment='right')
            n += 1

    ax1.set_ylabel('step length q1 (deg)', color='blue')
    ax2.set_ylabel('steering q2 (1)', color='red')
    ax1.set_xlabel('steps (1)')
    ax2.tick_params('y', colors='red')
    ax1.tick_params('y', colors='blue')
    
    save.save_plt_as_tikz('Out/opt_pathplanner/Q1Q2.tex')

    # %%
    plt.figure('Q2')
    plt.plot(Q2)
    plt.xlabel('steps (1)')
    plt.ylabel('steering q2 (1)')

    # %%
    plt.figure('DIST')
    plt.plot(DIST)
    plt.xlabel('steps (1)')
    plt.ylabel('dist to goal (cm)')




    # %% Animation

    #    gait.animate()
