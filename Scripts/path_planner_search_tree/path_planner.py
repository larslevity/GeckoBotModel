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
    
    XREF = [(0, 0), (20, 30), (-45, 50), (20, 95)]
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
#            alpha = add_noise(alpha)
            
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
        lab = 'goal '+str(idx) if idx > 0 else 'start'
        st.draw_point_dir(xref, [0, 0], msize=15, label=lab, fontsize=30)
    st.draw_point_arrow(p1, [np.cos(np.deg2rad(eps0))*10, np.sin(np.deg2rad(eps0))*10],
                             size=15, colp='orange')
    plt.grid()
    plt.xticks([-45, 0, 20.001], ['-45', '0', '20'])
    plt.yticks([0.001, 30, 50, 95], ['0', '30', '50', '95'])
    plt.ylim((-25, 120))
    plt.xlim((-25, 110))
    
    plt.xlabel('$x$ position (cm)')
    plt.ylabel('$y$ position (cm)')
    
    plt.axis('scaled')
    
    plt.grid()
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.grid()

    kwargs = {'extra_axis_parameters':
              {'x=.1cm', 'y=.1cm', 'anchor=origin', 'xmin=-55',
               'xmax=37','axis line style={draw opacity=0}',
               'ymin=-20, ymax=105', 'tick pos=left',}}

    gait_str = gait.get_tikz_repr(dashed=0)
    save.save_plt_as_tikz('Out/pathplanner/gait.tex', gait_str,
                          scope='scale=.1', **kwargs)
    
# %%

#    gait.animate()
