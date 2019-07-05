# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:11:13 2018

@author: AmP
"""

import numpy as np
from scipy.optimize import minimize


f_len = 100.     # factor on length objective
f_ori = 0.001  # .0003     # factor on orientation objective
f_ang = 10     # factor on angle objective

blow = .9       # lower stretching bound
bup = 1.1       # upper stretching bound

dev_angle = 100  # allowed deviation of angles

arc_res = 40    # resolution of arcs


class RobotRepr(object):
    def __init__(self, f_len=f_len, f_ori=f_ori, f_ang=f_ang, debug=False):
        self.cost = {'f_len': f_len, 'f_ori': f_ori, 'f_ang': f_ang}
        self.len_leg = 1
        self.len_tor = 1.1
        self.ref = {'alp1': 0.1, 'alp2': .1,
                    'gam': .1, 'bet1': .1, 'bet2': .1}
        self.state = {'alp1': 0.1, 'alp2': .1,
                      'gam': 0.1, 'bet1': .1, 'bet2': .1,
                      'F1': True, 'F2': False, 'F3': False, 'F4': False}
        self.meta = {'C1': 90, 'C2': -90, 'C3': 90, 'C4': -90, 'eps': 90,
                     'C0': {'C1': 90, 'C2': -90, 'C3': 90, 'C4': -90},
                     'l1': self.len_leg, 'l2': self.len_leg,
                     'lg': self.len_tor, 'l3': self.len_leg,
                     'l4': self.len_leg}
        self.coords = {'F1': (0, 0), 'F2': (2*self.len_leg, 0),
                       'F3': (0, -self.len_tor),
                       'F4': (2*self.len_leg, -self.len_tor),
                       'OM': (self.len_leg, 0),
                       'UM': (self.len_leg, -self.len_tor)}
        self.debug = debug
        self.calc_pose()
        self.state['F4'] = True
        self.calc_pose()

    def set_pose(self, pose):
        """ pose = (alp1, bet1, gam, alp2, bet2, F1, F2, F3, F4) """
        alp1, bet1, gam, alp2, bet2, F1, F2, F3, F4 = pose
        for a in [alp1, bet1, gam, alp2, bet2]:
            if a == 0:
                a = .01
        self.ref['alp1'] = alp1
        self.ref['alp2'] = alp2
        self.ref['gam'] = gam
        self.ref['bet1'] = bet1
        self.ref['bet2'] = bet2
        self.state['F1'] = F1
        self.state['F2'] = F2
        self.state['F3'] = F3
        self.state['F4'] = F4
        _, objective = self.calc_pose()
        return objective

    def set_initial_pose(self, alp1, bet1, gam, alp2, bet2, eps):
        for a in [alp1, bet1, gam, alp2, bet2]:
            if a == 0:
                a = .01
        self.meta['eps'] = eps
        self.ref['alp1'] = alp1
        self.ref['alp2'] = alp2
        self.ref['gam'] = gam
        self.ref['bet1'] = bet1
        self.ref['bet2'] = bet2
        self.state['F1'] = True
        self.state['F2'] = False
        self.state['F3'] = False
        self.state['F4'] = False
        self.coords['F1'] = (0, 0)
        self.calc_pose()

    def get_coords(self):
        print '\n'
        print 'Coords F1', self.coords['F1']
        print 'Coords F2', self.coords['F2']
        print 'Coords F3', self.coords['F3']
        print 'Coords F4', self.coords['F4']
        print '\n'

    def tikz_interface(self):
        F1 = self.coords['F1']
        alpha = [self.state[key] for key in
                 ['alp1', 'bet1', 'gam', 'alp2', 'bet2']]
        ell = [self.meta[key] for key in ['l1', 'l2', 'lg', 'l3', 'l4']]
        eps = self.meta['eps']
        return alpha, ell, eps, F1

    def get_repr(self):
        eps = self.meta['eps']
        c1 = self.meta['C1']
        l1 = self.meta['l1']
        l2 = self.meta['l2']
        l3 = self.meta['l3']
        l4 = self.meta['l4']
        lg = self.meta['lg']
        alp1 = self.state['alp1']
        bet1 = self.state['bet1']
        alp2 = self.state['alp2']
        bet2 = self.state['bet2']
        gam = self.state['gam']
        x1, y1 = self.coords['F1']
        x2, y2 = self.coords['F2']
        x3, y3 = self.coords['F3']
        x4, y4 = self.coords['F4']
        xf, yf = [x1, x2, x3, x4],  [y1, y2, y3, y4]
        xom, yom = self.coords['OM']
        xum, yum = self.coords['UM']

        x, y = [x1], [y1]
        # draw upper left leg
        xl1, yl1 = calc_arc_coords((x1, y1), c1, c1+alp1,
                                   calc_rad(l1, alp1))
        x = x + xl1
        y = y + yl1
        # draw torso
        xt, yt = calc_arc_coords((x[-1], y[-1]), -90+c1+alp1, -90+c1+alp1+gam,
                                 calc_rad(lg, gam))
        x = x + xt
        y = y + yt
        # draw lower right leg
        xl4, yl4 = calc_arc_coords((x[-1], y[-1]), -90+gam-(90-c1-alp1),
                                   -90+gam-(90-c1-alp1)-bet2,
                                   calc_rad(l4, bet2))
        x = x + xl4
        y = y + yl4
        # draw upper right leg
        xl2, yl2 = calc_arc_coords((xom, yom), c1+alp1,
                                   c1+alp1+bet1,
                                   calc_rad(l2, bet1))
        x = x + xl2
        y = y + yl2
        # draw lower left leg
        xl3, yl3 = calc_arc_coords((xum, yum), -90+gam-(90-c1-alp1),
                                   -90+gam-(90-c1-alp1)+alp2,
                                   calc_rad(l3, alp2))
        x = x + xl3
        y = y + yl3

        feet = ['F1', 'F2', 'F3', 'F4']
        fp = ([], [])
        nfp = ([], [])
        for idx, foot in enumerate(feet):
            if self.state[foot]:
                fp[0].append(xf[idx])
                fp[1].append(yf[idx])
            else:
                nfp[0].append(xf[idx])
                nfp[1].append(yf[idx])

        return (x, y), fp, nfp, eps

    def calc_pose(self):
        len_leg = self.len_leg
        len_tor = self.len_tor
        x1, y1 = self.coords['F1']
        x4, y4 = self.coords['F4']
        c0 = [self.meta['C1'], self.meta['C2'],
              self.meta['C3'], self.meta['C4']]
        eps0 = self.meta['eps']
        l10 = len_leg
        l20 = len_leg
        lg0 = len_tor
        l30 = len_leg
        l40 = len_leg
        alp10 = self.ref['alp1']
        bet10 = self.ref['bet1']
        gam0 = self.ref['gam']
        alp20 = self.ref['alp2']
        bet20 = self.ref['bet2']

        # Initial guess
        X0 = np.array([eps0, l10, l20, lg0, l30, l40,
                       alp10, bet10, gam0, alp20, bet20])

        def objective(X):
            f_len = self.cost['f_len']
            f_ori = self.cost['f_ori']
            f_ang = self.cost['f_ang']
            eps, l1, l2, lg, l3, l4, alp1, bet1, gam, alp2, bet2 = X
            c1 = np.mod(eps - alp1 - gam*.5 + 360, 360)
            c2 = np.mod(c1 + alp1 + bet1 + 360, 360)
            c3 = np.mod(180 + gam - bet1 + alp2 + c2 + 360, 360)
            c4 = np.mod(180 + gam + alp1 - bet2 + c1 + 360, 360)
            C = [c1, c2, c3, c4]
            feet = ['F1', 'F2', 'F3', 'F4']
            obj_ori = 0
            for idx, foot in enumerate(feet):
                if self.state[foot]:
                    obj_ori = (obj_ori + (C[idx] - c0[idx])**2)
#                    obj_ori = (obj_ori + calc_difference(C[idx], c0[idx])**2)

            obj_ang = 0
            for key in self.ref:
                obj_ang = obj_ang + (self.ref[key]-self.state[key])**2
#                obj_ang = obj_ang + calc_difference(self.ref[key], self.state[key])**2

            objective = (f_ori*np.sqrt(obj_ori) +
                         f_ang*np.sqrt(obj_ang) +
                         f_len*np.sqrt((l1-len_leg)**2 + (l2-len_leg)**2 +
                                       (lg-len_tor)**2 +
                                       (l3-len_leg)**2 + (l4-len_leg)**2))
            return objective

        def constraint1(X):
            """ feet should be at the right position """
            _, _, (xf1, yf1), (xf2, yf2), (xf3, yf3), (xf4, yf4) = \
                self.calc_coords(X)
            xf = [xf1, xf2, xf3, xf4]
            yf = [yf1, yf2, yf3, yf4]
            feet = ['F1', 'F2', 'F3', 'F4']
            constraint = 0
            for idx, foot in enumerate(feet):
                if self.state[foot]:
                    constraint = constraint + \
                        np.sqrt((self.coords[foot][0] - xf[idx])**2 +
                                (self.coords[foot][1] - yf[idx])**2)
            return constraint

        bleg = (blow*len_leg, bup*len_leg)
        btor = (blow*len_tor, bup*len_tor)
        bang = [(self.ref[ang]-dev_angle, self.ref[ang]+dev_angle)
                for ang in ['alp1', 'bet1', 'gam', 'alp2', 'bet2']]
        bnds = ((-360, 360), bleg, bleg, btor, bleg, bleg,
                bang[0], bang[1], bang[2], bang[3], bang[4])
        con1 = {'type': 'eq', 'fun': constraint1}
        cons = ([con1])
        solution = minimize(objective, X0, method='SLSQP',
                            bounds=bnds, constraints=cons)
        X = solution.x
        eps, l1, l2, lg, l3, l4, alp1, bet1, gam, alp2, bet2 = X
        c1 = np.mod(eps - alp1 - gam*.5 + 360, 360)
        c2 = np.mod(c1 + alp1 + bet1 + 360, 360)
        c3 = np.mod(180 + gam - bet1 + alp2 + c2 + 360, 360)
        c4 = np.mod(180 + gam + alp1 - bet2 + c1 + 360, 360)
        (xom, yom), (xum, yum), (xf1, yf1), (xf2, yf2), \
            (xf3, yf3), (xf4, yf4) = self.calc_coords(X)
        # save opt meta data
        if self.debug:
            print 'constraint function: ', constraint1(X)
            print 'objective function: ', objective(X)
            print 'solution vector: ', X

        self.meta['C1'] = c1
        self.meta['C2'] = c2
        self.meta['C3'] = c3
        self.meta['C4'] = c4
        self.meta['eps'] = eps
        self.meta['l1'] = l1
        self.meta['l2'] = l2
        self.meta['l3'] = l3
        self.meta['l4'] = l4
        self.meta['lg'] = lg
        self.state['alp1'] = alp1
        self.state['alp2'] = alp2
        self.state['gam'] = gam
        self.state['bet1'] = bet1
        self.state['bet2'] = bet2
        self.coords['OM'] = (xom, yom)
        self.coords['UM'] = (xum, yum)
        self.coords['F1'] = (xf1, yf1)
        self.coords['F2'] = (xf2, yf2)
        self.coords['F3'] = (xf3, yf3)
        self.coords['F4'] = (xf4, yf4)

        return X, objective(X)

    def calc_coords(self, X):
        """ X = [c1, l1, lg, l4]"""
        eps, l1, l2, lg, l3, l4, alp1, bet1, gam, alp2, bet2 = X
        c1 = eps - alp1 - gam*.5
        c2 = eps + bet1 - gam*.5
        r1 = calc_rad(l1, alp1)
        rg = calc_rad(lg, gam)
        r4 = calc_rad(l4, bet2)
        r3 = calc_rad(l3, alp2)
        r2 = calc_rad(l2, bet1)

        if self.state['F1']:
            xf1, yf1 = self.coords['F1']
            # coords cp upper left leg
            xr1 = xf1 + np.cos(np.deg2rad(c1))*r1
            yr1 = yf1 + np.sin(np.deg2rad(c1))*r1
            # coords upper torso
            xom = xr1 - np.sin(np.deg2rad(90-c1-alp1))*r1
            yom = yr1 - np.cos(np.deg2rad(90-c1-alp1))*r1
            # coords cp R2
            xr2 = xom + np.cos(np.deg2rad(c1+alp1))*r2
            yr2 = yom + np.sin(np.deg2rad(c1+alp1))*r2
            # coords F2
            xf2 = xr2 + np.sin(np.deg2rad(bet1 - (90-c1-alp1)))*r2
            yf2 = yr2 - np.cos(np.deg2rad(bet1 - (90-c1-alp1)))*r2
        elif self.state['F2']:
            xf2, yf2 = self.coords['F2']
            # coords cp upper right leg
            xr2 = xf2 - np.sin(np.deg2rad(c2-90))*r2
            yr2 = yf2 + np.cos(np.deg2rad(c2-90))*r2
            # coords upper torso
            xom = xr2 - np.sin(np.deg2rad(90-c2+bet1))*r2
            yom = yr2 - np.cos(np.deg2rad(90-c2+bet1))*r2
            # coords cp R1
            xr1 = xom + np.sin(np.deg2rad(90-c2+bet1))*r1
            yr1 = yom + np.cos(np.deg2rad(90-c2+bet1))*r1
            # coords F1
            xf1 = xr1 - np.sin(np.deg2rad(90-c2+bet1+alp1))*r1
            yf1 = yr1 - np.cos(np.deg2rad(90-c2+bet1+alp1))*r1

        # coords cp torso
        xrom = xom + np.cos(np.deg2rad(90-c1-alp1))*rg
        yrom = yom - np.sin(np.deg2rad(90-c1-alp1))*rg
        # coords lower torso
        xum = xrom - np.cos(np.deg2rad(gam - (90-c1-alp1)))*rg
        yum = yrom - np.sin(np.deg2rad(gam - (90-c1-alp1)))*rg
        # coords cp lower right foot
        xr4 = xum + np.sin(np.deg2rad(gam - (90-c1-alp1)))*r4
        yr4 = yum - np.cos(np.deg2rad(gam - (90-c1-alp1)))*r4
        # coords of F4
        xf4 = xr4 + np.sin(np.deg2rad(bet2 - (gam - (90-c1-alp1))))*r4
        yf4 = yr4 + np.cos(np.deg2rad(bet2 - (gam - (90-c1-alp1))))*r4
        # coords cp R3
        xr3 = xum + np.sin(np.deg2rad(gam - (90-c1-alp1)))*r3
        yr3 = yum - np.cos(np.deg2rad(gam - (90-c1-alp1)))*r3
        # coords of F3
        xf3 = xr3 - np.cos(np.deg2rad(-alp2 - gam + 180-c1-alp1))*r3
        yf3 = yr3 + np.sin(np.deg2rad(-alp2 - gam + 180-c1-alp1))*r3

        return ((xom, yom), (xum, yum), (xf1, yf1),
                (xf2, yf2), (xf3, yf3), (xf4, yf4))


def calc_len(radius, angle):
    return angle/360.*2*np.pi*radius


def calc_rad(length, angle):
    return 360.*length/(2*np.pi*angle)


def calc_arc_coords(xy, alp1, alp2, rad):
    """ """
    x0, y0 = xy
    x, y = [x0], [y0]
    xr = x0 + np.cos(np.deg2rad(alp1))*rad
    yr = y0 + np.sin(np.deg2rad(alp1))*rad
    steps = [angle for angle in np.linspace(0, alp2-alp1, arc_res)]
    for dangle in steps:
        x.append(xr - np.sin(np.deg2rad(90-alp1-dangle))*rad)
        y.append(yr - np.cos(np.deg2rad(90-alp1-dangle))*rad)

    return x, y


def calc_difference(phi0, phi1):
    theta = -np.radians(phi0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x2 = np.cos(np.radians(phi1))
    y2 = np.sin(np.radians(phi1))
    vec_ = R*np.c_[[x2, y2, 0]]
    diff = np.degrees(np.arctan2(float(vec_[1]), float(vec_[0])))
    return diff


if __name__ == "__main__":
    """
    To save the animation you need the libav-tool to be installed:
    sudo apt-get install libav-tools
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib
    matplotlib.use("Agg")

    save = False

    robrepr = RobotRepr()
    robrepr.meta['eps'] = 90
    robrepr.set_pose((.1, .1, .1, .1, .1, True, False, False, False))

    (x, y), fp, nfp = robrepr.get_repr()
    fpx, fpy = fp
    nfpx, nfpy = nfp
    plt.plot(x, y, 'k.')
    plt.plot(fpx, fpy, 'ko', markersize=10)
    plt.plot(nfpx, nfpy, 'kx', markersize=10)

    poses = []
#    for i in range(14):
#        poses.append((.1, 90, 90, .1, 90, False, True, True, False))
#        poses.append((.1, 90, 90, .1, 90, True, False, False, True))
#        poses.append((5, 45, 45, .1, 45, True, False, False, True))
#        poses.append((10, 0.1, -10, 10, .1, True, False, False, True))
#        poses.append((10, .1, -10, 10, .1, False, True, True, False))
#        poses.append((5, 45, 45, .1, 45, False, True, True, False))
    for i in range(11):
        poses.append((5, 90, 90, 5, 90, False, True, True, False))
        poses.append((5, 90, 90, 5, 90, True, False, False, True))
        poses.append((5, 45, 45, 5, 45, True, False, False, True))
        poses.append((10, 5, -10, 10, 5, True, False, False, True))
        poses.append((10, 5, -10, 10, 5, False, True, True, False))
        poses.append((5, 45, 45, 5, 45, False, True, True, False))

    data, data_fp, data_nfp, costs = [], [], [], []
    for idx, pose in enumerate(poses):
        print '\n\nPOSE ', idx, '\n'
        col = (.1, .5, float(idx)/len(poses))
        objj = robrepr.set_pose(pose)
        costs.append(objj)
        (x, y), fp, nfp = robrepr.get_repr()
        fpx, fpy = fp
        nfpx, nfpy = nfp
        plt.plot(x, y, '.', color=col)
        plt.plot(fpx, fpy, 'o', markersize=15, color=col)
        plt.plot(nfpx, nfpy, 'x', markersize=10, color=col)
        robrepr.get_coords()
        data.append((x, y))
        data_fp.append((fpx, fpy))
        data_nfp.append((nfpx, nfpy))

    plt.axis('equal')

    fig_obj = plt.figure()
    plt.title('Costs or "Stress" inside the robot')
    plt.plot(costs)

    # Animation
    def update_line(num, data, line, data_fp, line_fp, data_nfp, line_nfp):
        x, y = data[num]
        fpx, fpy = data_fp[num]
        nfpx, nfpy = data_nfp[num]
        line.set_data(np.array([[x], [y]]))
        line_fp.set_data(np.array([[fpx], [fpy]]))
        line_nfp.set_data(np.array([[nfpx], [nfpy]]))
        return line, line_fp, line_nfp

    fig1 = plt.figure()

    n = len(poses)
    l, = plt.plot([], [], '.')
    lfp, = plt.plot([], [], 'o', markersize=15)
    lnfp, = plt.plot([], [], 'x', markersize=10)
    plt.xlim(-4, 4)
    plt.ylim(-6, 2)
#    plt.axis('equal')
    plt.title('Gecko-robot model walking a circle')
    line_ani = animation.FuncAnimation(fig1, update_line, n,
                                       fargs=(data, l, data_fp, lfp,
                                              data_nfp, lnfp),
                                       interval=300, blit=True)

    if save:
        # Set up formatting for the movie files
        Writer = animation.writers['avconv']
        writer = Writer(fps=15, metadata=dict(artist='Lars Schiller'),
                        bitrate=1800)
        line_ani.save('circle.mp4', writer=writer)

    plt.show()
