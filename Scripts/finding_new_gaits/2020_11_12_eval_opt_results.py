#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:33:27 2020

@author: ls
"""

import pickle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as matcols
import numpy as np

import tikzplotlib


def load_obj(name):
    with open('Out/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
RESULTS = load_obj('RESULTS_curve')




def plot3dScatter(arr, cmap, F, name=''):
    fig = plt.figure(name)
    ax = fig.add_subplot(111, projection='3d')
    minval = np.min(arr)
    maxval = np.max(arr)
    
    for iori in range(len(arr)):
        for iang in range(len(arr[0])):
            for ilen in range(len(arr[0][0])):
                val = arr[iori][iang][ilen]
                [f_len, f_ang, f_ori] = np.log10(F[ilen][iang][iori])
                
                
                
                normval = (val-minval)/maxval
                col = cmap(normval)

                ax.scatter(f_len, f_ang, f_ori, marker='o', color=col,
                           s=5+normval*20)
#                ax.text(f_len, f_ang, f_ori, '%2.1f' % val, 'x')
   
    ax.set_xlabel('$f_l$')
    ax.set_ylabel('$f_a$')
    ax.set_zlabel('$f_o$')
    norm = matcols.Normalize(vmin=minval, vmax=maxval, clip=False)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)



def plot2dScatter(arr, cmap, F, name='', markers='linear' , **kwargs):
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    minval = np.min(arr)
    maxval = np.max(arr)
    
    N_flen = len(arr[0][0])
    
    for iori in range(len(arr)):
        for iang in range(len(arr[0])):
            for ilen in range(len(arr[0][0])):
                val = arr[iori][iang][ilen]
                [f_len, f_ang, f_ori] = (F[ilen][iang][iori])
                [f_len10, f_ang10, f_ori10] = np.log10(F[ilen][iang][iori])
                normval = (val-minval)/maxval
                col = cmap(normval)
                zscale= .12
                
                if markers == 'quadratic':
                    size = int(1+9*(-4*normval**2+4*normval))
                elif markers == 'linear':
                    size = int(1+9*normval)
                elif markers == 'exponetial_decrease':
                    size = int(10*10**(-normval*2))
                else:
                    size = int(10-5*normval)

                ax.plot([10**(f_ori10 +f_len10*zscale)], [10**(f_ang10+f_len10*zscale)],
                         color=col, markersize=size, **kwargs)
                if iori == 0 and iang == 0 and ilen in [0, N_flen-1]:
                    ax.annotate('$10^{%d}$' % f_len10, 
                                (10**(f_ori10 +f_len10*zscale-.2), 10**(f_ang10+f_len10*zscale+.1)))

    ax.annotate('$w_{\\ell}$' % f_len10, 
                (10**(-2-zscale+.2), 10**(-2-zscale-.2)))
        
    ax.annotate('', xytext=(10**(-2 +-2*zscale), 10**(-2+-2*zscale-.2)), 
                xy=(10**(-2 +3*zscale), 10**(-2+3*zscale-.2)),
                arrowprops=dict(facecolor='black', width=.01,
                                headwidth=3, headlength=4))
                    
                    
#                ax.text(f_len, f_ang, f_ori, '%2.1f' % val, 'x')

    ax.set_xlabel('$w_{\\varphi}$')
    ax.set_ylabel('$w_{\\alpha}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    norm = matcols.Normalize(vmin=minval, vmax=maxval, clip=False)
    bar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    bar.set_label(name)
    plt.grid()
    fig.tight_layout()
    return fig



              




for method in ['COBYLA']:
    print('METHOD:\t', method)
    for cat in RESULTS[method]:
        f_len = list(RESULTS[method][cat])[0]
        f_ang = list(RESULTS[method][cat][f_len])[0]
        N_flen = len(RESULTS[method][cat])
        N_fang = len(RESULTS[method][cat][f_len])
        N_fori = len(RESULTS[method][cat][f_len][f_ang])
        F = {}
        STRESS = np.zeros((N_fori, N_fang, N_flen))
        DX = np.zeros((N_fori, N_fang, N_flen))
        DY = np.zeros((N_fori, N_fang, N_flen))
        DEPS = np.zeros((N_fori, N_fang, N_flen))

        for ilen, f_len in enumerate(RESULTS[method][cat]):
            F[ilen] = {}
            for iang, f_ang in enumerate(RESULTS[method][cat][f_len]):
                F[ilen][iang] = {}
                for iori, f_ori in enumerate(RESULTS[method][cat][f_len][f_ang]):
                    F[ilen][iang][iori] = [f_len, f_ang, f_ori]
                    stress = RESULTS[method][cat][f_len][f_ang][f_ori]['stress']
                    dist = RESULTS[method][cat][f_len][f_ang][f_ori]['dist']
                    deps = RESULTS[method][cat][f_len][f_ang][f_ori]['deps']
                    STRESS[iori][iang][ilen] = stress
                    DX[iori][iang][ilen] = abs(dist[0])  # obacht!
                    DY[iori][iang][ilen] = dist[1]
                    DEPS[iori][iang][ilen] = abs(deps)
                    
                    alp_opt = RESULTS[method][cat][f_len][f_ang][f_ori]['alp']

        ampel = plt.get_cmap('RdYlGn')
        ampel_rev = plt.get_cmap('RdYlGn_r')
        twilight = plt.get_cmap('twilight')
        jet = plt.get_cmap('jet')
        coolwarm = plt.get_cmap('coolwarm')
        #%%
        fig = plot2dScatter(DY, coolwarm, F, '$\\Delta y$',
                      markers='linear', marker='o')
        fig.savefig('Out/'+method+'_'+cat+'_DY.pdf')
        

        fig=plot2dScatter(DX, ampel_rev, F, '$\\Delta x$', markers='exponetial_decrease', marker='o')
        fig.savefig('Out/'+method+'_'+cat+'_DX.pdf')
        fig=plot2dScatter(STRESS, ampel_rev, F, 'stress', markers='decrease', marker='o')
        fig.savefig('Out/'+method+'_'+cat+'_STRESS.pdf')
        fig=plot2dScatter(DEPS, ampel_rev, F, '$\\Delta \\varepsilon$', markers='exponetial_decrease', marker='o')
        fig.savefig('Out/'+method+'_'+cat+'_DEPS.pdf')

#        plot3dScatter(DEPS, ampel_rev, F, 'Delta epsilon')
        
        
        plt.show()
        
                 
                    

                    
