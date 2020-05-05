# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:20:14 2019

@author: debian
"""


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    import sys
    from os import path
    sys.path.insert(0, path.dirname(path.dirname(path.dirname(
            path.abspath(__file__)))))

    from Src.TrajectoryPlanner import state_machine as st
    from Src.Utils import plot_fun as pf
    from Src.Utils import save
    from Src.Math import kinematic_model as model

    alpha = [90, 0, -90, 90, 0]
    eps0 = 90
    p1 = (0, 0)
    ell = [9.1, 9.1, 10.3, 9.1, 9.1]
    f_l, f_o, f_a = 1, 1, 1



    x, (mx, my), f = model.set_initial_pose(alpha, eps0, p1, len_leg=ell[0], len_tor=ell[2])
    initial_pose = pf.GeckoBotPose(x, (mx, my), f,
                                   len_leg=ell[0], len_tor=ell[2])
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    ref = st.ReferenceGenerator('L')
    xref = (20, 30)

    def calc_dist(pose, xref):
        mx, my = pose.markers
        act_pos = np.r_[mx[1], my[1]]
        dpos = xref - act_pos
        return np.linalg.norm(dpos)

    def add_noise(alpha):
        return list(np.r_[alpha]+np.random.normal(0, 5, 5))

    i = 0
    
    XREF = [(20, 30), (-50, 50), (20, 100)]
    for xref in XREF:
        while calc_dist(gait.poses[-1], xref) > 8:
            act_pose = gait.poses[-1]
            x, y = act_pose.markers
            act_pos = (x[1], y[1])
            eps = act_pose.x[-1]
    
            alpha, feet, _,  pose_id = ref.get_next_reference(
                    act_pos, eps, xref, act_pose, save_as_tikz=True, gait=gait)
            print('\n\npose ', i)
            print('pose:\t\t', pose_id, ' -- ', alpha)
            print('distance:\t', calc_dist(gait.poses[-1], xref))
            # NOISE
            alpha = add_noise(alpha)
            
            if ':' not in pose_id or pose_id == 'crawling':
                predicted_pose = model.predict_next_pose(
                        [alpha, feet], act_pose.x, (x, y),
                        len_leg=ell[0], len_tor=ell[2], f=[f_l, f_o, f_a])
        
                predicted_pose = pf.GeckoBotPose(*predicted_pose)
                gait.append_pose(predicted_pose)
            i += 1
            if i > 200:
                break


# %%

    gait.plot_markers(1)
#    gait.plot_com()
    for idx, xref in enumerate(XREF):
        st.draw_point_dir(xref, [0, 0], msize=15, label='goal '+str(idx))
    st.draw_point_arrow(p1, [np.cos(np.deg2rad(eps0))*10, np.sin(np.deg2rad(eps0))*10],
                             size=15, label='goal '+str(idx), colp='orange')
    plt.axis('off')


    gait_str = gait.get_tikz_repr(dashed=0)
    save.save_plt_as_tikz('Out/pathplanner/gait.tex', gait_str,
                          scope='scale=.1')
    
# %%

#    gait.animate()
