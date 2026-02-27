############################################
#### Script to generate plots for the visualization of landscapes
############################################


import numpy as np
import time
import os
import datetime
import scipy.interpolate as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.tri as tri
from scipy.interpolate import RegularGridInterpolator


assemble = True ### if set to true assembles a file with all the landscape data
assemble = False

timecode = 'final_spring' ### file code for landscape data

if assemble: 
    vals = []
    vals_g = []
    vals_ad = []
    E = []
    for jj in range(32):   
        code = timecode + str(jj)
        filename_ls = './LS/' + code + '_Regular_Landscape.npy'
        filename_ls_g = './LS/' + code + '_Glob_Landscape.npy'
        filename_ls_ad = './LS/' + code + '_Adap_Landscape.npy'
        data = np.load(filename_ls, allow_pickle = True)
        data_g = np.load(filename_ls_g, allow_pickle = True)
        data_ad = np.load(filename_ls_ad, allow_pickle = True)
        vals = np.append(vals, data['vals'])
        vals_g = np.append(vals_g, data_g['vals_stoch'])
        vals_ad = np.append(vals_ad, data_ad['vals_adap'])
        E = np.append(E, data['eigs'])
    x_1 = data['x_1']
    x_2 = data['x_2']

    filename_complete = './LS/' + timecode + '_Smooth_Landscape.npy'
    with open(filename_complete, 'wb') as f:
        np.savez(f, vals = vals, vals_g = vals_g, vals_ad = vals_ad, x_1 = x_1, x_2 = x_2, E =E)

#### load example run data 
pathing = './ExAS_' + 'final_new' ### file code for run data 
pp = 93 ### run id for example run to plot
filename_stoch = pathing +'/Run_'+ str(pp) + '_results_stoch.npy'
filename_hybrid = pathing +'/Run_'+ str(pp) + '_results_hyb.npy'
f_stoch = np.load(filename_stoch)
f_hyb = np.load(filename_hybrid)
sol1 = f_stoch['sol2'][:]
sol3 = f_hyb['sol3'][:]

### load landscape data
filename_complete = './LS/'+ timecode + '_Smooth_Landscape.npy'
data = np.load(filename_complete, allow_pickle = True)
x_1 = data['x_1']
x_2 = data['x_2']
vals = data['vals']
vals_g = data['vals_g']
vals_ad = data['vals_ad']
E = data['E'] 

num_samples = len(x_1[0])
print('resolution:', num_samples, ' x ', num_samples)


#### landscape GSM
cividis_new = plt.colormaps['viridis']
cmap = ListedColormap(cividis_new(np.linspace(0, 1, 256)))

fig2,ax2 = plt.subplots(nrows = 1)           
CS2 = ax2.contourf(x_1, x_2, np.asarray(vals).reshape(num_samples,num_samples), cmap = cmap, levels = 40, zorder = 1)
cbar = fig2.colorbar(CS2)
ax2.set_xlabel(r"$\mathbf{x}[1]$")
ax2.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0, labelpad = 10)
ax2.yaxis.set_label_coords(-0.12,0.47)
fig2.suptitle(r'Landscape of $f$')

fig2g,ax2g = plt.subplots(nrows = 1)           
CS2g = ax2g.contourf(x_1, x_2, np.asarray(vals_g).reshape(num_samples,num_samples), cmap = cmap, levels = 40, zorder = 1)
cbar = fig2g.colorbar(CS2g)
ax2g.set_xlabel(r"$\mathbf{x}[1]$")
ax2g.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0, labelpad = 10)
ax2g.yaxis.set_label_coords(-0.12,0.47)
fig2g.suptitle(r'Landscape of $F$')

ax2g.scatter(sol1[:,0], sol1[:,1], c ='aqua', label = 'GSM', marker = 's')
ax2g.scatter(sol1[0,0],sol1[0,1],c = 'magenta', label = 'Starting Point')
ax2g.scatter(sol1[-1,0], sol1[-1,1],c = 'red', label = 'Minimum' )
ax2g.legend(loc = 'upper left')
fig2g.suptitle(r'Landscape of $F$ and Optimization Steps')


### landscape ASM
fig2ad,ax2ad = plt.subplots(nrows = 1)           
CS2ad = ax2ad.contourf(x_1, x_2, np.asarray(vals_ad).reshape(num_samples,num_samples), cmap = cmap, levels = 40, zorder = 1)
cbar = fig2ad.colorbar(CS2ad)
ax2ad.set_xlabel(r"$\mathbf{x}[1]$")
ax2ad.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0, labelpad = 10)
ax2ad.yaxis.set_label_coords(-0.12,0.47)
fig2ad.suptitle(r'Landscape of $\mathbb{F}$')

interp_E = RegularGridInterpolator((x_1[:,0], x_2[0,:]), E.reshape(num_samples,num_samples))

ind_3 = np.where(interp_E(sol3)**2  < 0.7)
ind_0 = np.setdiff1d(range(len(sol3[:,0])), ind_3)    
ax2ad.scatter(sol3[ind_0,0], sol3[ind_0,1], c = 'yellow', label = 'ASM' ,  marker = 'D')
ax2ad.scatter(sol3[0,0],sol3[0,1],c = 'magenta', label = 'Starting Point')
ax2ad.scatter(sol3[ind_3,0], sol3[ind_3,1], c = 'darkorange', label = 'Smoothing is applied', marker = '<' )
ax2ad.scatter(sol3[-1,0], sol3[-1,1],c = 'red', label = 'Minimum')
ax2ad.legend(loc = 'upper left')
ax2ad.set_xlabel(r'$\mathbf{x}[1]$')
ax2ad.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0, labelpad = 10)
ax2ad.yaxis.set_label_coords(-0.12,0.47)
fig2ad.suptitle(r'Landscape of $ \mathbb{F}$ and Optimization Steps')

E_sq = E**2
levels = [0.5, 0.7]  # thresholds

# Draw on the same axes as mathbb{F} landscape
CS_boundaries = ax2ad.contour(
    x_1, x_2, np.asarray(E_sq).reshape(num_samples, num_samples),
    levels=levels,
    colors=['red','orange'],
    linestyles=['--', '--'],
    linewidths=1,
    zorder=3
)

### landscape G
fig2b,ax2b = plt.subplots(nrows = 1)           
CS2 = ax2b.contourf(x_1, x_2, np.asarray(E).reshape(num_samples,num_samples), cmap = cmap, levels = 40, zorder = 1)
cbar = fig2b.colorbar(CS2)
ax2b.set_xlabel(r"$\mathbf{x}[1]$")
ax2b.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0, labelpad = 10)
ax2b.yaxis.set_label_coords(-0.12,0.47)
fig2b.suptitle(r'Landscape of $G$')

E_sq = E**2
levels = [0.5, 0.7]  # thresholds

# Draw on the same axes as G landscape
CS_boundaries = ax2b.contour(
    x_1, x_2, np.asarray(E_sq).reshape(num_samples, num_samples),
    levels=levels,
    colors=['red','orange'],
    linestyles=['--', '--'],
    linewidths=1,
    zorder=3
)

plt.show()

