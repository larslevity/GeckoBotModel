# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 16:43:14 2019

@author: AmP
"""


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    from os import path
    sys.path.insert(0, path.dirname(path.dirname(path.dirname(
            path.abspath(__file__)))))

    from Src.TrajectoryPlanner import search_tree as st
    from Src.Utils import plot_fun as pf
    from Src.Utils import save
    from Src.Math import kinematic_model as model

    alpha = [90, 0, -90, 90, 0]
    eps = 90
    p1 = (0, 0)
    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    ref = st.ReferenceGenerator('010')
    xref = (2, 3)

    def add_noise(alpha):
        return list(np.r_[alpha]+np.random.normal(0, 5, 5))

    def calc_dist(pose, xref):
        mx, my = pose.markers
        act_pos = np.r_[mx[1], my[1]]
        dpos = xref - act_pos
        return np.linalg.norm(dpos)

    i = 0
    while calc_dist(gait.poses[-1], xref) > .5:
        act_pose = gait.poses[-1]
        x, y = act_pose.markers
        act_pos = (x[1], y[1])
        eps = act_pose.x[-1]

        alpha, feet, _,  pose_id = ref.get_next_reference(
                act_pos, eps, xref, act_pose, save_as_tikz=True, gait=gait,
                save_png=False)
#        alpha = add_noise(alpha)
        print(alpha)
        print(calc_dist(gait.poses[-1], xref))

        predicted_pose = model.predict_next_pose(
                [alpha, feet], act_pose.x, (x, y))

        predicted_pose = pf.GeckoBotPose(*predicted_pose)
        gait.append_pose(predicted_pose)
        i += 1
        if i > 100:
            break


# %%
    gait.plot_gait()
    gait.plot_markers(1)
#    gait.plot_com()
    st.draw_point_dir(xref, [0, 0], size=20, label='GOAL1')
    plt.axis('off')

#    plt.savefig('Out/pathplanner/gait.png', transparent=False,
#                dpi=300)
    plt.show()
    plt.close('GeckoBotGait')

# %% Tikz Pic

    gait.plot_markers(1)
    st.draw_point_dir(xref, [0, 0], size=20, label='GOAL1')
    plt.axis('off')
    gait_str = gait.get_tikz_repr()
#    save.save_plt_as_tikz('Out/pathplanner/gait.tex', gait_str)

# %%
    gait.animate()
