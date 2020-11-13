#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:06:32 2020

@author: ls
"""

import pickle

import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
from os import path
sys.path.insert(0, path.dirname(path.dirname(path.dirname(
        path.abspath(__file__)))))

from Src.Utils import plot_fun as pf
from Src.Math import kinematic_model as model
from Src.Utils import save


def gnerate_ptrn(X, n_cycles, half=False):
    ptrn = []
    for n in range(n_cycles):
        p = X
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
        ptrn.append([[p[5], p[6], p[7], p[8], p[9]], [0, 1, 1, 0]])
    if half:
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
    return ptrn


def mirror(p):
    return [p[1], p[0], -p[2], p[4], p[3]]


def gnerate_ptrn_symmetric(X, n_cycles, half=False):
    ptrn = []
    for n in range(n_cycles):
        p = X
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
        ptrn.append([[p[1], p[0], -p[2], p[4], p[3]], [0, 1, 1, 0]])
    if half:
        ptrn.append([[p[0], p[1], p[2], p[3], p[4]], [1, 0, 0, 1]])
    return ptrn


def optimize_gait_straight(n_cycles, f_weights, method='COBYLA',
                           x0=[90, .1, -90, 90, .1]):
    obj_history = []
    bleg = (0.1, 120)
    btor = (-120, 120)
    bnds = [bleg, bleg, btor, bleg, bleg]
    X0 = x0

    def objective_straight(X):
        x, marks, _ = model.set_initial_pose(
            X, 90, (0, 0), len_leg=1, len_tor=1.2)

        ptrn = gnerate_ptrn_symmetric(X, n_cycles)
        for ref in ptrn:
            x, marks, _, _, _ = model.predict_next_pose(
                    ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
            mx, my = marks

        obj = -my[1]
        obj_history.append(obj)
        print('step', len(obj_history), '\t', round(obj, 4))
        return obj

    solution = minimize(objective_straight, X0, method=method, bounds=bnds,
                        tol=1e-1)
    X = solution.x
    ptrn = gnerate_ptrn_symmetric(X, n_cycles)
    return ptrn, obj_history, objective_straight(X)


def optimize_gait_curve(n_cycles, f_weights, method='COBYLA',
                        x0=[90, .1, -90, 90, .1]):
    obj_history = []
    bleg = (0.01, 120)
    btor = (-120, 120)
    bnds = [bleg, bleg, btor, bleg, bleg, bleg, bleg, btor, bleg, bleg]
    X0 = model.flat_list([x0, x0])

    def objective_curve(X):
        x, marks, _ = model.set_initial_pose(
            x0, 90, (0, 0), len_leg=1, len_tor=1.2)

        ptrn = gnerate_ptrn(X, n_cycles)

        for ref in ptrn:
            x, marks, _, _, _ = model.predict_next_pose(
                    ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
            eps = x[-1]

        obj = 90-eps
        obj_history.append(obj)
        print('step', len(obj_history), '\t', round(obj, 4))
        return obj

    solution = minimize(objective_curve, X0, method=method, bounds=bnds,
                        tol=1e-2)
    X = solution.x
    ptrn = gnerate_ptrn(X, n_cycles)
    return ptrn, obj_history, objective_curve(X)



def save_obj(obj, name):
    with open('Out/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('Out/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# %% MAIN
#if __name__ == '__main__':
    
#f_len = 100.     # factor on length objective
#f_ori = 10.  # .0003     # factor on orientation objective
#f_ang = 10.     # factor on angle objective

n_cycles = 1
#cat = 'curve'
#cat = 'straight'


FLEN = [0.01, 0.1, 1, 10, 100, 1000]
FANG = [0.01, 0.1, 1, 10, 100, 1000]
FORI = [0.01, 0.1, 1, 10, 100, 1000]
methods = ['COBYLA', 'SLSQP'] # ['Powell', 'COBYLA', 'CG', 'TNC', 'SLSQP']
x0 = [90, .1, -90, 90, .1]


RESULTS = {}


for method in methods:
    RESULTS[method] = {}
    for cat in ['curve']:  #, 'straight']:
        RESULTS[method][cat] = {}
        for f_len in FLEN:
            RESULTS[method][cat][f_len] = {}
            for f_ang in FANG:
                RESULTS[method][cat][f_len][f_ang] = {}
                for f_ori in FORI:
                    RESULTS[method][cat][f_len][f_ang][f_ori] = {}
                    f_weights = [f_len, f_ori, f_ang]
                    
                    
        
                    # STRAIGHT
                    if cat == 'straight':
                        opt_ptrn, obj_hist, opt_obj = optimize_gait_straight(
                                n_cycles, f_weights=f_weights, method=method, x0=x0)
        
                    # CURVE
                    if cat == 'curve':
                        opt_ptrn, obj_hist, opt_obj = optimize_gait_curve(
                                n_cycles, f_weights=f_weights, method=method, x0=x0)
        
                    # %% PRINT STATS
                    if cat == 'straight':
                        alp_opt = [round(ref, 2) for ref in opt_ptrn[0][0]]
        
                        init_pose = [opt_ptrn[0][0], 90, (0, 0)]
                        prop_str = '{}_{}_{}_{}_{}_{}_{}__{}_{}_{}_{}_{}'.format(
                                method, n_cycles,
                                f_len, f_ori, f_ang,
                                len(obj_hist), round(opt_obj, 2),
                                alp_opt[0], alp_opt[1], alp_opt[2], alp_opt[3], alp_opt[4])
        
                    if cat == 'curve':
                        alp_opt = [round(ref, 2) for ref in opt_ptrn[0][0]]
                        alp_opt += [round(ref, 2) for ref in opt_ptrn[1][0]]
        
                        init_pose = [opt_ptrn[0][0], 90, (0, 0)]
                        prop_str = '{}_{}_{}_{}_{}_{}_{}__{}_{}_{}_{}_{}__{}_{}_{}_{}_{}'.format(
                                method, n_cycles,
                                f_len, f_ori, f_ang,
                                len(obj_hist), round(opt_obj, 2),
                                alp_opt[0], alp_opt[1], alp_opt[2], alp_opt[3], alp_opt[4],
                                alp_opt[5], alp_opt[6], alp_opt[7], alp_opt[8], alp_opt[9])
        
                    print('initial guess:\n', x0, '\n')
                    print('property string:\n', prop_str, '\n')
                    print('optimal pattern:\n', alp_opt, '\n')
        
                    # %% RENDER GAIT & PLOTS
        
                    plt.figure('opt_hist: ' + prop_str)
                    plt.title('opt_hist: ' + prop_str)
                    plt.plot(obj_hist)
        
                    x, marks, f = model.set_initial_pose(
                        alp_opt[:5], 90, (0, 0), len_leg=1, len_tor=1.2)
                    initial_pose = pf.GeckoBotPose(x, marks, f)
                    gait = pf.GeckoBotGait()
                    gait.append_pose(initial_pose)
                    if cat == 'straight':
                        ptrn = gnerate_ptrn_symmetric(alp_opt, 1, half=True)
                    if cat == 'curve':
                        ptrn = gnerate_ptrn(alp_opt, 1, half=True)
                    for ref in ptrn:
                        x, marks, f, constraint, cost = model.predict_next_pose(
                                ref, x, marks, len_leg=1, len_tor=1.2, f=f_weights)
                        gait.append_pose(
                                pf.GeckoBotPose(x, marks, f, constraint=constraint, cost=cost))
                    
                    stress = gait.plot_stress()
                    dist, deps = gait.get_travel_distance()
                    
                    RESULTS[method][cat][f_len][f_ang][f_ori]['stress'] = stress
                    RESULTS[method][cat][f_len][f_ang][f_ori]['dist'] = dist
                    RESULTS[method][cat][f_len][f_ang][f_ori]['deps'] = deps
                    RESULTS[method][cat][f_len][f_ang][f_ori]['alp'] = alp_opt
                    
                    save_obj(RESULTS, 'RESULTS_curve')
        
                    gait.plot_gait()
                    gait.plot_markers([1])
                    plt.axis('off')
                    plt.savefig('Out/opt_ref/'+cat+'/{}.png'.format(prop_str),
                                transparent=False, dpi=150)
#                    plt.show()
                    plt.close('all')
        
                    # %% SAVE AS TIKZ
                    gait.plot_markers(1)
                    plt.axis('off')
                    gait_str = gait.get_tikz_repr()
                    save.save_plt_as_tikz('Out/opt_ref/'+cat+'/{}.tex'.format(prop_str),
                                          gait_str)
                    plt.close('GeckoBotGait')


