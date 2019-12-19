#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:38:10 2019

@author: ls
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm


def calc_dx_dy_deps(X1, X2, RESULT_DEPS, RESULT_DX, RESULT_DY):

    X1__, X2__ = np.meshgrid(X1, X2)
    X1_ = X1__.flatten()
    X2_ = X2__.flatten()

    order = 2
    roundon = 3

    Adic = {}
    Adic[0] = [X1_*0+1]
    Adic[1] = [X1_, X2_]
    Adic[2] = [X1_**2, X2_**2, X1_*X2_]
    Adic[3] = [X1_**3, X2_**3, X1_**2*X2_, X2_**2*X1_]
    Adic[4] = [X1_**4, X2_**4, X1_**3*X2_**1, X1_**2*X2_**2, X1_**1*X2_**3]
    Adic[5] = [X1_**5, X2_**5, X1_**4*X2_**1, X1_**3*X2_**2, X1_**2*X2_**3, X1_**1*X2_**4]

    Tdic = {}
    Tdic[0] = '{}'
    Tdic[1] = ' + {}x_1^1 + {}x_2^1'
    Tdic[2] = ' + {}x_1^2 + {}x_2^2 + {}x_1^1x_2^1'
    Tdic[3] = ' + {}x_1^3 + {}x_2^3 + {}x_1^2x_2^1 + {}x_1^1x_2^2'
    Tdic[4] = ' + {}x_1^4 + {}x_2^4 + {}x_1^3x_2^1 + {}x_1^2x_2^2 + {}x_1^1x_2^3'
    Tdic[5] = ' + {}x_1^5 + {}x_2^5 + {}x_1^4x_2^1 + {}x_1^3x_2^2 + {}x_1^2x_2^3 + {}x_1^1x_2^4'

    pys = (
            '{}'
            + '+ {}*x1 + {}*x2'
            + '+ {}*x1**2 + {}*x2**2 + {}*x1**1*x2**1'
            + '+ {}*x1**3 + {}*x2**3 + {}*x1**2*x2**1 + {}*x1**1*x2**2'
            + '+ {}*x1**4 + {}*x2**4 + {}*x1**3*x2**1 + {}*x1**2*x2**2 + {}*x1**1*x2**3'
#                + '+ {}*x1**5 + {}*x2**5 + {}*x1**4*x2**1 + {}*x1**3*x2**2 + {}*x1**2*x2**3 + {}*x1**1*x2**4'
            )

    tex = ''
    for i in range(order+1):
        tex += Tdic[i]

    def flat_list(l):
        return [item for sublist in l for item in sublist]

    A = [Adic[i] for i in range(order+1)]
    A = flat_list(A)
    A = np.array(A).T

    ######################################################################
    # Plot DEPS
    # Plot the surface.
    levels = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]
    fig = plt.figure('Delta Epsilon')
    surf = plt.contourf(X2__.T, X1__.T, RESULT_DEPS.T, cmap=cm.coolwarm,
                        levels=levels)
    plt.colorbar()
    surf = plt.contour(X2__.T, X1__.T, RESULT_DEPS.T, levels=levels,
                       cmap=cm.coolwarm)
    # Add a color bar which maps values to colors.
    plt.clim(-100, 100)
    
    
#        fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.ylabel('step length $q_1$')
    plt.xlabel('steering $q_2$')

    # FIT DEPS

    B = RESULT_DEPS.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=-1)
    coeff_ = [round(c, roundon) for c in coeff]
    FIT = X1_*0
    for c, a in zip(coeff_, A.T):
        FIT += c*a
    FIT = np.reshape(FIT, np.shape(X1__), order='C')
    surf = plt.contour(X2__.T, X1__.T, FIT.T, '--', levels=levels, colors='k')
    ax1 = plt.gca()
#        ax1.clabel(surf, levels, inline=True, fmt='%2.0f')

    print(coeff_)
    fig = plt.gcf()
    deps = tex.format(*coeff_)
    plt.title('$\\delta \\varepsilon='+deps+'$')
    plt.gca().invert_yaxis()
    fig.set_size_inches(10.5, 8.5)
    fig.savefig('../../Out/analytic_model7/FitDeps_{}_order_{}_rounded_{}.png'.format(order, roundon),
                dpi=300, trasperent=True, bbox_inches='tight')

    ######################################################################
    # Plot DXDY
    fig, ax = plt.subplots(num='DXDY')

    BDX = RESULT_DX.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, BDX)
    coeff_ = [round(c, roundon) for c in coeff]

    FITDX = X1_*0
    for c, a in zip(coeff_, A.T):
        FITDX += c*a

    dx = tex.format(*coeff_)

    BDY = RESULT_DY.flatten()
    coeff, r, rank, s = np.linalg.lstsq(A, BDY)
    coeff_ = [round(c, roundon) for c in coeff]
    dy = tex.format(*coeff_)

    FITDY = X1_*0
    for c, a in zip(coeff_, A.T):
        FITDY += c*a

    error_x = np.reshape(((BDX - FITDX)), np.shape(X1__), order='C')
    error_y = np.reshape(((BDY - FITDY)), np.shape(X1__), order='C')
    error_len_abs = np.sqrt((error_x**2 + error_y**2))
    error_len_rel = error_len_abs / np.reshape(np.sqrt(BDX**2 + BDY**2), np.shape(X1__)) * 100

    mean_error = round(np.mean(np.nanmean(error_len_rel, 0)[1:]), 2)  # x1=0 excluded
#        error_len_rel[error_len_rel == np.inf] = 0
#        error_len_rel[np.isnan(error_len_rel)] = 0
    levels = [0, 5, 10, 15, 20, 25, 30, 50]
    contour = plt.contourf(X2__.T, X1__.T, error_len_rel.T, cmap='OrRd',
                           levels=levels)
    plt.colorbar(label='$|FIT - SIM|/|SIM|$ (\%), mean = {}\%'.format(mean_error))
#        plt.contour(contour, levels=levels)  #, colors='k')
#        plt.clabel(contour, levels, inline=True)  #, colors='k')
    plt.clim(0, 30)

    # PLOT VECTOR FIELD
    scale = 15
    M = np.hypot(RESULT_DX.T, RESULT_DY.T)
    FITDX = np.reshape(FITDX, np.shape(X1__))
    FITDY = np.reshape(FITDY, np.shape(X1__))
    q = ax.quiver(X2__.T, X1__.T, RESULT_DX.T, RESULT_DY.T, M, units='x', scale=scale)
    ax.scatter(X2__.T, X1__.T, color='0.5', s=10)
    ax.quiver(X2__.T, X1__.T, FITDX.T, FITDY.T, units='x', scale=scale, color='black')
    ax.quiver(X2__.T, X1__.T, error_x.T, error_y.T, units='x', scale=scale, color='red')
    ax.grid()

    print('dx =' + dx)
    print('dy =' + dy)

    tit = '$\delta x(x_1, x_2)= \\begin{array}{c} %s \\\ %s \\end{array}$' % (dx, dy)
    plt.title(tit)
    plt.xlabel('steering $q_2$')
    plt.ylabel('step length $q_1$')

    plt.gca().invert_yaxis()
    fig = plt.gcf()
    fig.set_size_inches(10.5, 8.5)
    fig.savefig('../../Out/analytic_model7/FitDXDY_{}_order_{}_round_{}.png'.format(order, roundon),
                dpi=300, trasperent=True, bbox_inches='tight')