# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:38:56 2019

@author: ls
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path
import os

try:
    from os import scandir
except ImportError:
    from scandir import scandir

from Src.Math import kinematic_model as model
from Src.Utils import save as mysave

n_limbs = 5
n_foot = 4
arc_res = 40    # resolution of arcs


class GeckoBotPose(object):
    def __init__(self, x, marks, f, constraint=0, cost=0,
                 len_leg=1, len_tor=1.2, name=None):
        self.x = x
        self.markers = marks
        self.f = f
        self.constraint = constraint
        self.cost = cost
        self.len_leg = len_leg
        self.len_tor = len_tor
        self.alp = self.x[0:n_limbs]
        self.ell = self.x[n_limbs:2*n_limbs]
        self.eps = self.x[-1]
        self.name = name

    def get_eps(self):
        return self.x[-1]

    def get_m1_pos(self):
        mx, my = self.markers
        return (mx[1], my[1])

    def plot(self, col='k'):
        (x, y), (fpx, fpy), (nfpx, nfpy) = \
            get_point_repr(self.x, self.markers, self.f)
        plt.plot(x, y, '.', color=col)
        plt.plot(fpx, fpy, 'o', markersize=10, color=col)
        plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)
        plt.axis('equal')

    def show_stats(self):
        alp, ell, eps = (self.x[0:n_limbs], self.x[n_limbs:2*n_limbs],
                         self.x[-1])
        phi = model._calc_phi(alp, eps)
        mx, my = self.markers
        print('constraint function: \t', round(self.constraint, 2))
        print('objective function: \t', round(self.cost, 2))
        print('alp: \t\t\t', [round(xx, 2) for xx in alp])
        print('ell: \t\t\t', [round(xx, 2) for xx in ell])
        print('mx: \t\t\t', [round(xx, 2) for xx in mx])
        print('my: \t\t\t', [round(xx, 2) for xx in my])
        print('phi: \t\t\t', [round(xx, 2) for xx in phi])
        print('eps: \t\t\t', round(eps, 2), '\n')

    def get_tikz_repr(self, col='black', xshift=None, linewidth='.7mm',
                      yshift=None, rotate=None, **kwargs):
        alp, ell, eps = (self.x[0:n_limbs], self.x[n_limbs:2*n_limbs],
                         self.x[-1])
        mx, my = self.markers
        geckostring = ''
        if xshift:
            geckostring += '\\begin{scope}[xshift=%scm]\n' % str(round(xshift, 4))
        if yshift:
            geckostring += '\\begin{scope}[yshift=%scm]\n' % str(round(yshift, 4))
        if rotate:
            geckostring += '\\begin{scope}[rotate=%s]\n' % str(round(rotate, 4))

        geckostring += tikz_draw_gecko(
                alp, ell, eps, (mx[0], my[0]), fix=self.f, col=col,
                linewidth=linewidth, posename=self.name, **kwargs)
        if xshift:
            geckostring += '\\end{scope}\n'
        if yshift:
            geckostring += '\\end{scope}\n'
        if rotate:
            geckostring += '\\end{scope}\n'
        geckostring += '\n\n'
        return geckostring

    def save_as_tikz(self, filename, compileit=True):
        direc = path.dirname(path.dirname(path.dirname(
                    path.abspath(__file__)))) + '/Out/'
        gstr = self.get_tikz_repr()
        name = direc+filename+'.tex'
        mysave.save_geckostr_as_tikz(name, gstr)
        if compileit:
            out_dir = os.path.dirname(name)
            print(name)
            os.system('pdflatex -output-directory {} {}'.format(out_dir, name))
            print('Done')

    def get_phi(self):
        alp, eps = (self.x[0:n_limbs], self.x[-1])
        phi = model._calc_phi(alp, eps)
        return phi

    def get_alpha(self):
        return self.x[0:n_limbs]


class GeckoBotGait(object):
    def __init__(self, initial_pose=None):
        self.poses = []
        if initial_pose:
            self.append_pose(initial_pose)

    def append_pose(self, pose):
        self.poses.append(pose)

    def plot_gait(self, fignum='', figname='GeckoBotGait', reverse_col=0):
        plt.figure(figname+fignum)
        for idx, pose in enumerate(self.poses):
            c = (1-float(idx)/len(self.poses))*.8
            if reverse_col:
                c = 1 - c
            col = (c, c, c)
            pose.plot(col)

    def get_tikz_repr(self, shift=None, reverse_col=0, linewidth='.7mm',
                      **kwargs):
        gait_str = ''
        for idx, pose in enumerate(self.poses):
            c = int(20 + (float(idx)/len(self.poses))*80.)
            if reverse_col:
                c = 120 - c
            if reverse_col < 0:
                c = abs(reverse_col)
            col = 'black!{}'.format(c)
            shift_ = idx*shift if shift else None
            gait_str += pose.get_tikz_repr(col, shift_, linewidth, **kwargs)
        return gait_str

    def plot_markers(self, markernum=range(6), figname='GeckoBotGait'):
        """plots the history of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure(figname)
        marks = [pose.markers for pose in self.poses]
        markers = marker_history(marks)
        col = markers_color()
        for idx, marker in enumerate(markers):
            if idx in markernum:
                x, y = marker
                plt.plot(x, y, color=col[idx])
        plt.axis('equal')

    def plot_com(self, markernum=range(6)):
        """plots the history of center of markers in *markernum*"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure('GeckoBotGait')
        marks = [pose.markers for pose in self.poses]
        markers = marker_history(marks)
        x = np.zeros(len(markers[0][0]))
        y = np.zeros(len(markers[0][0]))
        for idx, marker in enumerate(markers):
            if idx in markernum:
                xi, yi = marker
                x = x + np.r_[xi]
                y = y + np.r_[yi]
        x = x/len(markernum)
        y = y/len(markernum)
        plt.plot(x, y, color='purple')
        plt.axis('equal')

    def plot_markers2(self, markernum=range(6)):
        """
        plots every value of markers in *markernum*
        if torso is bent positive"""
        if type(markernum) == int:
            markernum = [markernum]
        plt.figure('GeckoBotGait')
        marks = []
        for pose in self.poses:
            if pose.x[2] > 0:
                marks.append(pose.markers)
        markers = marker_history(marks)
        col = markers_color()
        for idx, marker in enumerate(markers):
            if idx in markernum:
                x, y = marker
                plt.plot(x, y, color=col[idx])
        plt.axis('equal')

    def get_travel_distance(self):
        last = self.poses[-1].get_m1_pos()
        start = self.poses[0].get_m1_pos()
        dist = (last[0]-start[0], last[-1]-start[1])
        deps = self.poses[-1].get_eps() - self.poses[0].get_eps()
        return dist, deps

    def plot_travel_distance(self, shift=[0, 0], colp='orange', size=12, w=.8):
        plt.figure('GeckoBotGait')
        dist, deps = self.get_travel_distance()
        start = self.poses[0].get_m1_pos()

        plt.plot([start[0]+shift[0]], [start[1]+shift[1]], marker='o', color=colp, markersize=size)
        plt.plot([start[0]+shift[0], start[0]+shift[0]+dist[0], start[0]],
                 [start[1]+shift[1], start[1]+shift[1]+dist[1], start[1]], alpha=0)
        plt.arrow(start[0]+shift[0], start[1]+shift[1], dist[0], dist[1], color='blue', width=w,
                  length_includes_head=True)

    def plot_orientation(self, length=1, poses=[0, -1], shift=[0,0], colp='k', w=.1, size=12):
        plt.figure('GeckoBotGait')
        for pose in poses:
            start = self.poses[pose].get_m1_pos()
            eps = self.poses[pose].get_eps()
            plt.plot([start[0]+shift[0], start[0]+shift[0]+np.cos(np.deg2rad(eps))*length],
                     [start[1]+shift[1], start[1]+shift[1]+np.sin(np.deg2rad(eps))*length], color=colp, alpha=0)
            plt.plot([start[0]+shift[0]], [start[1]+shift[1]], marker='o', color=colp, markersize=size)
            plt.arrow(start[0]+shift[0], start[1]+shift[1],
                      np.cos(np.deg2rad(eps))*length,
                      np.sin(np.deg2rad(eps))*length, color=colp, width=w,
                      length_includes_head=True)

    def plot_epsilon(self):
        plt.figure('GeckoBotGaitEpsHistory')
        Eps = []
        for pose in self.poses:
            eps = pose.get_eps()
            Eps.append(eps)
        plt.plot(Eps, 'purple')
        return Eps

    def plot_phi(self):
        Phi = [[], [], [], []]
        for pose in self.poses:
            phi = pose.get_phi()
            for idx, phii in enumerate(phi):
                Phi[idx].append(phii)
        plt.figure('GeckoBotGaitPhiHistory')
        col = markers_color()
        for idx, phi in enumerate(Phi):
            plt.plot(phi, color=col[idx])
        plt.legend(['0', '1', '2', '3'])
        return Phi

    def plot_alpha(self):
        Alp = [[], [], [], [], []]
        for pose in self.poses:
            alp = pose.get_alpha()
            for idx, alpi in enumerate(alp):
                Alp[idx].append(alpi)
        plt.figure('GeckoBotGaitAlphaHistory')
        col = markers_color()
        for idx, alp in enumerate(Alp):
            plt.plot(alp, color=col[idx])
        plt.legend(['0', '1', '2', '3', '4'])
        return Alp

    def plot_stress(self, fignum=''):
        plt.figure('GeckoBotGaitStress'+fignum)
        stress = [pose.cost for pose in self.poses]
        plt.plot(stress)
        return sum(stress)

    def save_as_tikz(self, filename, latexcompile=False):
        direc = path.dirname(path.dirname(path.dirname(
                    path.abspath(__file__))))
        name = direc+filename+'.tex'
        out_dir = os.path.dirname(name)
        gstr = ''
        for idx, pose in enumerate(self.poses):
            if len(self.poses) == 1:
                col = 'black'
            else:
                c = 50 + int((float(idx)/(len(self.poses)-1))*50)
                col = 'black!{}'.format(c)
            gstr += pose.get_tikz_repr(col)
        mysave.save_geckostr_as_tikz(name, gstr)
        if latexcompile:
            os.system('pdflatex -output-directory {} {}'.format(out_dir, name))
        print('Done')

    def animate(self):
        fig1 = plt.figure('GeckoBotGaitAnimation')
        data_xy, data_markers = [], []
        for pose in self.poses:
            (x, y), (fpx, fpy), (nfpx, nfpy) = \
                get_point_repr(pose.x, pose.markers, pose.f)
            data_xy.append((x, y))
            data_markers.append(pose.markers)
        line_ani = animate_gait(fig1, data_xy, data_markers)
        plt.show('GeckoBotGaitAnimation')
        return line_ani


def predict_gait(references, initial_pose, weight=None, lens=[None]):
    if not lens:
        lens = [1, 1.2]
    len_leg = lens[0]
    len_tor = lens[1]
    if not weight:
        weight = [model.f_l, model.f_o, model.f_a]

    gait = GeckoBotGait(initial_pose)
    for idx, ref in enumerate(references):
        x, (mx, my), f, constraint, cost = model.predict_next_pose(
                ref, gait.poses[idx].x, gait.poses[idx].markers,
                f=weight, len_leg=len_leg, len_tor=len_tor)
        gait.append_pose(GeckoBotPose(
                x, (mx, my), f, constraint=constraint,
                cost=cost))
    _ = gait.poses.pop(0)  # remove initial pose
    return gait


def markers_color():
    return ['red', 'orange', 'green', 'blue', 'magenta', 'darkred']


def get_actuator_tikzcolor():
    return ['red', 'red!50!black', 'orange', 'blue', 'blue!50!black']


def get_point_repr(x, marks, f):
    alp, ell, eps = x[0:n_limbs], x[n_limbs:2*n_limbs], x[-1]
    c1, _, _, _ = model._calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    alp1, bet1, gam, alp2, bet2 = alp
    xf, yf = model.get_feet_pos(marks)

    x, y = [xf[0]], [yf[0]]
    # draw upper left leg
    xl1, yl1 = _calc_arc_coords((x[-1], y[-1]), c1, c1+alp1,
                                model._calc_rad(l1, alp1))
    x = x + xl1
    y = y + yl1
    # draw torso
    xt, yt = _calc_arc_coords((x[-1], y[-1]), -90+c1+alp1, -90+c1+alp1+gam,
                              model._calc_rad(lg, gam))
    x = x + xt
    y = y + yt
    # draw lower right leg
    xl4, yl4 = _calc_arc_coords((x[-1], y[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)-bet2,
                                model._calc_rad(l4, bet2))
    x = x + xl4
    y = y + yl4
    # draw upper right leg
    xl2, yl2 = _calc_arc_coords((xl1[-1], yl1[-1]), c1+alp1,
                                c1+alp1+bet1, model._calc_rad(l2, bet1))
    x = x + xl2
    y = y + yl2
    # draw lower left leg
    xl3, yl3 = _calc_arc_coords((xt[-1], yt[-1]), -90+gam-(90-c1-alp1),
                                -90+gam-(90-c1-alp1)+alp2,
                                model._calc_rad(l3, alp2))
    x = x + xl3
    y = y + yl3

    fp = ([], [])
    nfp = ([], [])
    for idx in range(n_foot):
        if f[idx]:
            fp[0].append(xf[idx])
            fp[1].append(yf[idx])
        else:
            nfp[0].append(xf[idx])
            nfp[1].append(yf[idx])

    return (x, y), fp, nfp


def _calc_arc_coords(xy, alp1, alp2, rad):
    x0, y0 = xy
    x, y = [x0], [y0]
    xr = x0 + np.cos(np.deg2rad(alp1))*rad
    yr = y0 + np.sin(np.deg2rad(alp1))*rad
    steps = [angle for angle in np.linspace(0, alp2-alp1, arc_res)]
    for dangle in steps:
        x.append(xr - np.sin(np.deg2rad(90-alp1-dangle))*rad)
        y.append(yr - np.cos(np.deg2rad(90-alp1-dangle))*rad)

    return x, y


def marker_history(marks):
    """ formats the marks from predictpose to:
        marks[marker_idx][x/y][pose_idx]
    """
    markers = [([], []), ([], []), ([], []), ([], []), ([], []), ([], [])]
    for pose in range(len(marks)):
        x, y = marks[pose]
        for xm, ym, idx in zip(x, y, range(len(x))):
            markers[idx][0].append(xm)
            markers[idx][1].append(ym)
    return markers


def extract_eps(data):
    (data_, data_fp, data_nfp, data_x) = data
    eps = []
    for pose_idx in range(len(data_x)):
        eps.append(data_x[pose_idx][-1])
    return eps


def extract_ell(data):
    (data_, data_fp, data_nfp, data_x) = data
    ell = []
    for pose_idx in range(len(data_x)):
        ell.append(data_x[pose_idx][n_limbs:2*n_limbs])
    return ell


def plot_gait(data_xy, data_fp, data_nfp, data_x):
    for idx in range(len(data_xy)):
        (x, y) = data_xy[idx]
        (fpx, fpy) = data_fp[idx]
        (nfpx, nfpy) = data_nfp[idx]
        c = (1-float(idx)/len(data_xy))*.8
        col = (c, c, c)
        plt.plot(x, y, '.', color=col)
        plt.plot(fpx, fpy, 'o', markersize=10, color=col)
        plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)
    plt.axis('equal')


def start_end(data_xy, data_fp, data_nfp, data_x):
    return ([data_xy[0], data_xy[-1]], [data_fp[0], data_fp[-1]],
            [data_nfp[0], data_nfp[-1]], [data_x[0], data_x[-1]])


def start_mid_end(data_xy, data_fp, data_nfp, data_x):
    mid = len(data_xy)/2
    return ([data_xy[0], data_xy[mid], data_xy[-1]],
            [data_fp[0], data_fp[mid], data_fp[-1]],
            [data_nfp[0], data_nfp[mid], data_nfp[-1]],
            [data_x[0], data_x[mid], data_x[-1]])


def animate_gait(fig1, data_xy, data_markers, inv=500,
                 col=['red', 'orange', 'green', 'blue', 'magenta', 'darkred']):

    def update_line(num, data_xy, line_xy, data_markers,
                    lm0, lm1, lm2, lm3, lm4, lm5, leps):
        x, y = data_xy[num]
        line_xy.set_data(np.array([[x], [y]]))

        xm0, ym0 = data_markers[num][0][0], data_markers[num][1][0]
        xm1, ym1 = data_markers[num][0][1], data_markers[num][1][1]
        xm2, ym2 = data_markers[num][0][2], data_markers[num][1][2]
        xm3, ym3 = data_markers[num][0][3], data_markers[num][1][3]
        xm4, ym4 = data_markers[num][0][4], data_markers[num][1][4]
        xm5, ym5 = data_markers[num][0][5], data_markers[num][1][5]
        lm0.set_data(np.array([[xm0], [ym0]]))
        lm1.set_data(np.array([[xm1], [ym1]]))
        lm2.set_data(np.array([[xm2], [ym2]]))
        lm3.set_data(np.array([[xm3], [ym3]]))
        lm4.set_data(np.array([[xm4], [ym4]]))
        lm5.set_data(np.array([[xm5], [ym5]]))
        leps.set_data(np.array([[xm1, xm4], [ym1, ym4]]))
        return line_xy, lm0, lm1, lm2, lm3, lm4, lm5, leps

    n = len(data_xy)
    l_xy, = plt.plot([], [], 'k.', markersize=3)
    msize = 5
    lm0, = plt.plot([], [], 'o', color=col[0], markersize=msize)
    lm1, = plt.plot([], [], 'o', color=col[1], markersize=msize)
    lm2, = plt.plot([], [], 'o', color=col[2], markersize=msize)
    lm3, = plt.plot([], [], 'o', color=col[3], markersize=msize)
    lm4, = plt.plot([], [], 'o', color=col[4], markersize=msize)
    lm5, = plt.plot([], [], 'o', color=col[5], markersize=msize)
    leps, = plt.plot([], [], '-', color='mediumpurple', linewidth=1)

    minx, maxx, miny, maxy = 0, 0, 0, 0
    for dataset in data_markers:
        x, y = dataset
        minx = min(x) if minx > min(x) else minx
        maxx = max(x) if maxx < max(x) else maxx
        miny = min(y) if miny > min(y) else miny
        maxy = max(y) if maxy < max(y) else maxy
    plt.xlim(minx-5, maxx+5)
    plt.ylim(miny-5, maxy+5)

    line_ani = animation.FuncAnimation(
        fig1, update_line, n, fargs=(data_xy, l_xy, data_markers,
                                     lm0, lm1, lm2, lm3, lm4, lm5, leps),
        interval=inv, blit=True)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    return line_ani


def save_animation(line_ani, name='gait.mp4', conv='avconv'):
    """
    To save the animation you need the libav-tool to be installed:
    sudo apt-get install libav-tools

    To create gif:
        0. EASY: Use : https://ezgif.com/video-to-gif

        OR (hard way):

        1. Create a directory called frames in the same directory with
           your .mp4 file. Use command:
            ffmpeg -i video.mp4  -r 5 'frames/frame-%03d.jpg'

            -r 5 stands for FPS value
                for better quality choose bigger number
                adjust the value with the -delay in 2nd step
                to keep the same animation speed

            %03d gives sequential filename number in decimal form

        1a. Loop the thing (python):
            import os
            for jdx, idx in enumerate(range(1, 114)[::-1]):
                os.rename('frame-'+'{}'.format(idx).zfill(3)+'.jpg',
                          'frame-'+'{}'.format(114+jdx).zfill(3)+'.jpg')

        1b. Reduce size of single frames (bash):
            for i in *.jpg; do convert "$i" -quality 20 "${i%%.jpg*}_new.jpg"; done

        2. Convert Images to gif (bash):
            cd frames
            convert -delay 20 -loop 0 *.jpg myimage.gif

            -delay 20 means the time between each frame is 0.2 seconds
               which match 5 fps above.
               When choosing this value
                   1 = 100 fps
                   2 = 50 fps
                   4 = 25 fps
                   5 = 20 fps
                   10 = 10 fps
                   20 = 5 fps
                   25 = 4 fps
                   50 = 2 fps
                   100 = 1 fps
                   in general 100/delay = fps

            -loop 0 means repeat forever
        2a. To further compress you can skip frames:
            gifsicle -U input.gif `seq -f "#%g" 0 2 99` -O2 -o output.gif

    """
    # Set up formatting for the movie files
    Writer = animation.writers[conv]
    writer = Writer(fps=15, metadata=dict(artist='Lars Schiller'),
                    bitrate=1800)
    line_ani.save(name, writer=writer)


def tikz_draw_gecko(alp, ell, eps, F1, col='black', posename=None,
                    linewidth='.5mm', fix=None, dashed=1, R=.4):
    c1, c2, c3, c4 = model._calc_phi(alp, eps)
    l1, l2, lg, l3, l4 = ell
    for idx, a in enumerate(alp):
        if abs(a) < 2:
            alp[idx] = 2 * a/abs(a)
    alp1, bet1, gam, alp2, bet2 = alp
    r1, r2, rg, r3, r4 = [model._calc_rad(ell[i], alp[i]) for i in range(5)]
    if isinstance(col, str):
        col = [col]*5
    ls = ['', '', '', '']
    if fix and dashed:
        for idx in range(4):
            if not fix[idx]:
                ls[idx] = 'dashed, '

    elem = """
\\def\\lw{%s}
\\def\\alpi{%f}
\\def\\beti{%f}
\\def\\gam{%f}
\\def\\alpii{%f}
\\def\\betii{%f}
\\def\\gamh{%f}

\\def\\eps{%f}
\\def\\ci{%f}
\\def\\cii{%f}
\\def\\ciii{%f}
\\def\\civ{%f}

\\def\\ri{%f}
\\def\\rii{%f}
\\def\\rg{%f}
\\def\\riii{%f}
\\def\\riv{%f}

\\def\\R{%f}

\\path (%f, %f)coordinate(F1);

\\path (F1)arc(180+\\ci:180+\\ci+\\alpi:\\ri)coordinate(OM)arc(90+\\ci+\\alpi:90+\\ci+\\alpi+\\gam:\\rg)coordinate(UM);
\\path (OM)--(UM)node[midway, %s](middle){};

\\draw[%s, %s line width=\\lw] (F1)arc(180+\\ci:180+\\ci+\\alpi:\\ri)coordinate(OM);
\\draw[%s, %s line width=\\lw] (OM)arc(180+\\ci+\\alpi:180+\\ci+\\alpi+\\beti:\\rii)coordinate(F2);
\\draw[%s, line width=\\lw] (OM)arc(90+\\ci+\\alpi:90+\\ci+\\alpi+\\gam:\\rg)coordinate(UM);
\\draw[%s, %s line width=\\lw] (UM)arc(\\gam+\\ci+\\alpi:\\gam+\\ci+\\alpi+\\alpii:\\riii)coordinate(F3);
\\draw[%s, %s line width=\\lw] (UM)arc(\\gam+\\ci+\\alpi:\\gam+\\ci+\\alpi-\\betii:\\riv)coordinate(F4);

""" % (linewidth, alp1, bet1, gam, alp2, bet2, gam*.5, eps, c1, c2, c3, c4,
       r1, r2, rg, r3, r4, R, F1[0], F1[1], posename if posename else '',
       col[0], ls[0], col[1], ls[1], col[2], col[3], ls[2], col[4], ls[3])
    if fix:
        col_ = [col[0], col[1], col[3], col[4]]
        for idx, fixation in enumerate(fix):
            c = [c1, c2, c3, c4]
            if fixation:
                fixs = '\\draw[%s, line width=\\lw, fill] (F%s)++(%f :\\R) circle(\\R);\n' % (col_[idx], str(idx+1), 
                              c[idx]+90 if idx in [0, 3] else c[idx]-90)
                elem += fixs
            else:
                fixs = '\\draw[%s, line width=\\lw] (F%s)++(%f :\\R) circle(\\R);\n' % (col_[idx], str(idx+1),
                              c[idx]+90 if idx in [0, 3] else c[idx]-90)
                elem += fixs

    return elem
