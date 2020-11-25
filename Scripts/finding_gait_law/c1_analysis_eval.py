#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:57:54 2020

@author: ls
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as matcols
import numpy as np

import tikzplotlib


def load_obj(name):
    with open('Out/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


X1 = [50, 60, 70, 80, 90]
X2 = np.arange(-.5, .52, .2)
C1 = np.linspace(0, 2, 21)



def plot2dScatter(arr, cmap, name='', markers='linear', norm='total', **kwargs):
    

    fig = plt.figure(name)
    (cax, ax) = fig.subplots(2, 1, gridspec_kw={'height_ratios': [5, 95]})
#    ax = fig.add_subplot(121)
#    cax = fig.add_subplot(122, aspect=0.3)

#    N_x2 = len(X2)

#    RESULT_DX[x1_idx][x2_idx][c1_idx]

    if norm == 'total':        
        minval = np.min(arr)
        maxval = np.max(arr)


    for x1_idx in range(len(arr)):
        if norm != 'total':
            minval = np.min(arr[x1_idx])
            maxval = np.max(arr[x1_idx])

        for x2_idx in range(len(arr[0])):
            for c1_idx in range(len(arr[0][0])):
#                print(x1_idx, x2_idx, c1_idx)
                val = arr[x1_idx][x2_idx][c1_idx]
                x1 = X1[x1_idx]
                x2 = X2[x2_idx]
                c1 = C1[c1_idx]
                
                normval = (val-minval)/maxval
                
                col = cmap(normval)
                
                
                if markers == 'quadratic':
                    size = int(1+9*(-4*normval**2+4*normval))
                elif markers == 'linear':
                    normval = (val)/(maxval)
                    size = abs(10*normval)
                    if size < .5:
                        size = .5
                elif markers == 'exponetial_decrease':
                    size = int(10*10**(-normval*2))
                elif markers == 'exponetial':
                    size = (1+ 9-np.exp(-normval*4)*9)
                else:
                    size = int(10-5*normval)

                ax.plot([c1+x2*.2], [x1+x2*5],
                         color=col, markersize=size, **kwargs)
#                if c1_idx == 0 and x1_idx == 0 and x2_idx in [0, N_x2-1]:
#                    ax.annotate('${:.2}$'.format(x2), 
#                                (c1+x2*.2-.1, x1+x2*5+.1), fontsize=20)

    ax.annotate('$q_2$', 
                (0-.2, (50+2)), fontsize=20)
        
    ax.annotate('', xytext=((-.1), (50-2.5+1)), 
                xy=(0.1, 50+2.5+1),
                arrowprops=dict(facecolor='black', width=.01, headwidth=3, headlength=4))
                    

    ax.set_xlabel('additional bending $c_1~(1)$')
    ax.set_ylabel('step length $q_1~(^\\circ)$')
    ax.grid('on')
    
    # color bar hack to export as tikz
    y = np.linspace(0, 1, 2)
    if norm == 'total':
        x = np.linspace(minval, maxval, 100)
    else:
        x = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = X
    cax.contourf(X, Y, Z, cmap=cmap, levels=100)
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(name)
    
    cax.set_yticklabels([' '], color='white')
    cax.set_yticks([0])
    cax.yaxis.label.set_color('white')
    cax.tick_params(axis='y', colors='white')


    cax.grid('off')     

    fig.tight_layout()
    return fig


fname = 'c1_RESULTS[0.1, 10, 10]'

[RESULT_DX, RESULT_DY, RESULT_DEPS, RESULT_STRESS] = load_obj(fname)


## delet outlier 
RESULT_STRESS[4][0][19] = 300


ampel_rev = plt.get_cmap('RdYlGn_r')
coolwarm = plt.get_cmap('coolwarm')


fig=plot2dScatter(RESULT_STRESS, ampel_rev, 'normalized stress~$(1)$', markers='exponetial', norm=1,  marker='o')
tikzplotlib.save('Out/c1_analysis/'+fname+'STRESS.tex', standalone=True,
                 extra_axis_parameters={'width=8cm', 'height=8cm'},
                 extra_groupstyle_parameters={'vertical sep={0pt}'})

fig=plot2dScatter(RESULT_DEPS, coolwarm, '$\\Delta \\varepsilon~(^\\circ)$', markers='linear', marker='o')
tikzplotlib.save('Out/c1_analysis/'+fname+'DEPS.tex', standalone=True,
                 extra_axis_parameters={'width=8cm', 'height=8cm'},
                 extra_groupstyle_parameters={'vertical sep={0pt}'})


plt.show()