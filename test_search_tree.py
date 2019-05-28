# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:17:40 2019

@author: AmP
"""

if __name__ == "__main__":
    import numpy as np
    from Src.TrajectoryPlanner import search_tree as st
    from Src.Utils import plot_fun as pf
    from Src.Math import kinematic_model as model

    try:
        from graphviz import Digraph

        def render_graph(graph):
            """ requirements:
            pip install graphviz
            apt-get install graphviz
            """
            dot = Digraph()
            for v in graph.vertices():
                dot.node(v, v)
            for e in graph.edges():
                v = e[0]
                w = e[1]
                c = e[2]
                dot.edge(v, w, label=str(c) if c else None)
            dot.render('tree', view=True)

        graph = st.Graph(st.g)
        render_graph(graph)
    except ImportError:
        print('Missing package gaphiviz')
        print('run: "pip install graphviz" and "apt-get install graphviz" ')

    alpha = [90, 0, -90, 90, 0]
    eps = 90
    p1 = (0, 0)
    x, (mx, my), f = model.set_initial_pose(alpha, eps, p1)
    initial_pose = pf.GeckoBotPose(x, (mx, my), f)
    gait = pf.GeckoBotGait()
    gait.append_pose(initial_pose)

    ref = st.ReferenceGenerator('010')
    xref = (-12, -4)

    def calc_dist(pose, xref):
        mx, my = pose.markers
        act_pos = np.r_[mx[1], my[1]]
        dpos = xref - act_pos
        return np.linalg.norm(dpos)

    i = 0
    while calc_dist(gait.poses[-1], xref) > 1:
        act_pose = gait.poses[-1]
        x, y = act_pose.markers
        act_pos = (x[1], y[1])
        eps = act_pose.x[-1]
        alpha, feet, _,  pose_id = ref.get_next_reference(
                act_pos, eps, xref, act_pose)
        predicted_pose = model.predict_next_pose(
                [alpha, feet], act_pose.x, (x, y))
        predicted_pose = pf.GeckoBotPose(*predicted_pose)
        gait.append_pose(predicted_pose)
        i += 1
        if i > 12:
            break

    gait.plot_gait()
    st.draw_point_dir(xref, [0, 0], size=20, label='GOAL1')

    gait.animate()

