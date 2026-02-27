############################################
############################################


import numpy as np
import time
import os
import datetime
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.tri as tri


run_folder = './Ex_QSAF_final/' ### where example run is stored
run_id = 0 ### id of which run is plotted over the landscape

timecode = 'final_MS' ### code of data
suffix = '_04_02_32x32_MS_25_E_575_x2=175_sq' ### suffix of data


assemble = True ### set to true if you want to assemble from data, set false so nothing is overwritten by accident
assemble = False

if assemble: 
    vals = []
    for jj in range(64):   
        code = timecode + str(jj)
        filename_ls = './LS_ASM/' + code + '_Adap_Landscape' + suffix +'.npy' ### landscape files
        data = np.load(filename_ls, allow_pickle = True)
        vals = np.append(vals, data['vals_adap'])
    x_1 = data['x_1']
    x_2 = data['x_2']

    filename_complete = './LS_ASM/' + 'Complete_MS' + '_Adap_Landscape' + suffix +'.npy'
    with open(filename_complete, 'wb') as f:
        np.savez(f, vals = vals, x_1 = x_1, x_2 = x_2)

else:
    filename_complete = './LS_ASM/' + 'Complete_MS' + '_Adap_Landscape' + suffix +'.npy'  
    data = np.load(filename_complete, allow_pickle = True)
    x_1 = data['x_1']
    x_2 = data['x_2']
    vals = data['vals']

    num_samples = len(x_1[0])

    print('resolution:', num_samples, ' x ', num_samples)


    cividis_new = plt.colormaps['viridis']
    cmap = ListedColormap(cividis_new(np.linspace(0, 1, 256)))

    fig2,ax2 = plt.subplots(nrows = 1)           
    CS2 = ax2.contourf(x_1, x_2, np.asarray(vals).reshape(num_samples,num_samples), cmap = cmap, levels = 40, zorder = 1)
    cbar = fig2.colorbar(CS2)
    ax2.set_xlabel(r"$\mathbf{x}[1]$")
    ax2.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig2.suptitle(r'Landscape of $\mathbb{F}$')

    levels = [0.2**2, 0.4**2]
    
    CS_boundaries = ax2.contour(
        x_1, x_2, ((x_1-x_2)**2),
        levels=levels,
        colors=['red','orange'],
        linestyles=['--', '--'],
        linewidths=1.2,
        zorder=3
    )
    minen = np.argmin(vals)

    ax2.set_xlabel(r"$\mathbf{x}[1]$")
    ax2.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)

    #### load example run to plot over landscape
    ff= np.load(run_folder +'Run_'+ str(run_id) + '_results_hyb.npy', allow_pickle = True)
    sol3 = ff['sol3']

    ind_3 = np.where((sol3[:,0] - sol3[:,1])**2  < 0.4**2)
    ind_0 = np.setdiff1d(range(len(sol3[:,0])), ind_3)    
    ax2.scatter(sol3[ind_0,0], sol3[ind_0,1], c = 'yellow', label = 'ASM' ,  marker = 'D')
    ax2.scatter(sol3[0,0],sol3[0,1],c = 'magenta', label = 'Starting Point')
    ax2.scatter(sol3[ind_3,0], sol3[ind_3,1], c = 'darkorange', label = 'Smoothing is applied', marker = '<' )
    ax2.scatter(sol3[-1,0], sol3[-1,1],c = 'red', label = 'Minimum')
    ax2.legend(loc = 'upper left')
    

    print(x_1.shape)

    plt.show()

