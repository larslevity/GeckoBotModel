# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:18:10 2019

@author: AmP
"""
import numpy as np
from scipy.optimize import minimize


def rotate(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


def xbar(xref, xbot, epsbot):
    """ maps the reference point in global COS to robot COS """
    xref = np.array([[xref[0]], [xref[1]]])
    xbot = np.array([[xbot[0]], [xbot[1]]])
#    print(xref - xbot)
    return rotate(xref - xbot, np.deg2rad(-epsbot))


#def dx(x1, x2):  # Simulation Result
#    return np.array([
#            [.02*x1 + .13*x2 - .47*x2**2],
#            [-(.07*x2 - .29*x2**2 + .02*x1*x2)]])
#
#
#def deps(x1, x2):  # Simulation Results
#    return np.deg2rad(-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


def dx(x1, x2):  # Symmetric Fit
    return np.array([
            [.02*x1 + .13*abs(x2) - .47*x2**2],
            [-(.07*x2 - .29*x2**2*np.sign(x2) + .02*x1*(x2))]
        ])


def deps(x1, x2):  # Symmetric Fit
    return np.deg2rad(-.005*x1 - 10.85*x2 - 2.55*x2**2*np.sign(x2)
                      - .835*x1*x2)


def sumsin(x, n):
    return 1/np.sin(x/2)*np.sin((n+1)*x/2)*np.sin(n*x/2)


def sumcos(x, n):
    return 1/np.sin(x/2)*np.sin((n+1)*x/2)*np.cos(n*x/2)


def R(alp):
    return np.array(
            [[np.cos(alp), -np.sin(alp)],
             [np.sin(alp), np.cos(alp)]])


def sumR(alp, n):
    return np.array(
            [[sumcos(alp, n), -sumsin(alp, n)],
             [sumsin(alp, n), sumcos(alp, n)]
             ])


#def calc_d(xbar, dx, deps, n):  # Src
#    return np.linalg.norm(
#            R(-n*deps)*xbar - sumR(-deps, n)*dx)


def calc_d(xbar, dx, deps, n):  # Hack
    xbar = np.c_[xbar]
    xbar_n = np.matmul(R(-n*deps), xbar) - np.matmul(sumR(-deps, n), dx)

#    print('xbar:', xbar)
#    print('R(-n*deps):', R(-n*deps))
#    print('sumR(-deps, n):', sumR(-deps, n))
#    print('dx', dx)
#    print('xbar_n', xbar_n)
    d = np.linalg.norm(xbar_n)
    return d



def cut(x):
    return x if x > 0.001 else 0.001


def alpha(x1, x2, f):
    alpha = [cut(45 - x1/2. + (f[0] ^ 1)*x1*x2 + f[0]*abs(x1)*x2/2.),
             cut(45 + x1/2. + (f[1] ^ 1)*x1*x2 + f[1]*abs(x1)*x2/2.),
             x1 + x2*abs(x1),
             cut(45 - x1/2. + (f[2] ^ 1)*x1*x2 + f[2]*abs(x1)*x2/2.),
             cut(45 + x1/2. + (f[3] ^ 1)*x1*x2 + f[3]*abs(x1)*x2/2.)
             ]
    return alpha


def Jd(x1, x2, xbar, n, h=.001):
    d0 = calc_d(xbar, dx(x1, x2), deps(x1, x2), n)
    dx1 = calc_d(xbar, dx(x1+h, x2), deps(x1+h, x2), n)
    dx2 = calc_d(xbar, dx(x1, x2+h), deps(x1, x2+h), n)
    return np.array([(dx1 - d0)/h, (dx2 - d0)/h]), d0


def find_opt_x(xbar, n):
    def objective(x):
        x1, x2 = x
#        print([round(xx, 2) for xx in x])
        Jac, d = Jd(x1, x2, xbar, n)
#        print(round(d, 3))
        return d, Jac

    x0 = [90, 0]
    bnds = [(0, 90), (-.5, .5)]
    solution = minimize(objective, x0, method='L-BFGS-B', bounds=bnds,
                        jac=True, tol=1e-7)
    return solution.x


def optimal_planner(xbar, alp_act, feet_act, n=2, dist_min=.1, show_stats=0):
    """
    opt planner
    """
    dist_act = np.linalg.norm(xbar)
    if dist_act < dist_min:
        alp_ref = alp_act
        feet_ref = feet_act
        return [alp_ref, feet_ref]

    # Not the case -> lets optimize
    feet_ref = [not(foot) for foot in feet_act]
    x1opt, x2opt = find_opt_x(xbar, n)
    if alp_act[2] > 0:
        x1opt *= -1
    alpha_ref = alpha(x1opt, x2opt, feet_ref)

    if show_stats:
        print('x1: \t\t{}\nx2: \t\t{}'.format(round(x1opt, 2), round(x2opt, 2)))
#        print('R:', R(-n*deps)*xbar)

    return [alpha_ref, feet_ref]


if __name__ == "__main__":
    xref = [2, 3]  # global cos
    eps = 90
    p1 = (0, 0)
    n = 4
    xb = xbar(xref, p1, eps)
    feet0 = [1, 0, 0, 1]  # feet states
    alpha0 = [90, 0, -90, 90, 0]

    ref = optimal_planner(xb, alpha0, feet0, n=n)
    print(ref)


    # %% Analyze

    import matplotlib.pyplot as plt
    
    X1 = np.arange(0.01, 90.2, 10.)
    X2 = np.arange(-.511, .515, .1)

    RESULT_DX = np.zeros((len(X2), len(X1)))
    RESULT_DY = np.zeros((len(X2), len(X1)))
    RESULT_DEPS = np.zeros((len(X2), len(X1)))

    
    for x1_idx, x1 in enumerate(X1):
        for x2_idx, x2 in enumerate(X2):
            
            (dxx, dyy), ddeps = dx(x1, x2), deps(x1, x2)
            RESULT_DX[x2_idx][x1_idx] = dxx
            RESULT_DY[x2_idx][x1_idx] = dyy
            RESULT_DEPS[x2_idx][x1_idx] = ddeps
    
    
    
    # %% Plot DXDYDE
    X1__, X2__ = np.meshgrid(X1, X2)
    X1_ = X1__.flatten()
    X2_ = X2__.flatten()


     
    fig, ax = plt.subplots(num='DXDYDE')
    plt.title('DXDY, DE')
    
    levels = 15
    cset = ax.contourf(X1, X2, RESULT_DEPS, levels=levels, inline=1)
    cset = ax.contour(X1, X2, RESULT_DEPS, levels=levels, inline=1, colors='k')
    ax.clabel(cset, colors='k')

    M = np.hypot(RESULT_DX, RESULT_DY)
    q = ax.quiver(X1__, X2__, RESULT_DX, RESULT_DY, units='x', scale=.2)
    ax.scatter(X1__, X2__, color='0.5', s=10)


    ax.grid()


    # %% PLOT DE
    fig, ax = plt.subplots(num='DEPS')
    plt.title('DEPS')
    
    levels = 15
    cset = ax.contourf(X1, X2, RESULT_DEPS, levels=levels, inline=1)
    cset = ax.contour(X1, X2, RESULT_DEPS, levels=levels, inline=1, colors='k')
    ax.clabel(cset, colors='k')


    # %% PLOT DX
    fig, ax = plt.subplots(num='DX')
    plt.title('DX')
    
    levels = 15
    cset = ax.contourf(X1, X2, RESULT_DX, levels=levels, inline=1)
    cset = ax.contour(X1, X2, RESULT_DX, levels=levels, inline=1, colors='k')
    ax.clabel(cset, colors='k')

    # %% PLOT DY
    fig, ax = plt.subplots(num='DY')
    plt.title('DY')
    
    levels = 15
    cset = ax.contourf(X1, X2, RESULT_DY, levels=levels, inline=1)
    cset = ax.contour(X1, X2, RESULT_DY, levels=levels, inline=1, colors='k')
    ax.clabel(cset, colors='k')
