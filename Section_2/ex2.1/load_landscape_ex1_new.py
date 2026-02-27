############################################
#### Script to generate plots for the visualization of optimization runs
#### as well as the function landscapes for global and adaptively smoothing method.
#### "date_time" variable specifies the date for what run should be loaded.
#### "do_work" variable specifies, whether to generate new data for a landscape, or to load from a data file
#### "load_landscape" variable specifies whether to do extra plot of just function landscapes.
#### "Plot_landscape_smooth" variable specifies whether to do a plot for global smoothing method
#### "Plot_landscape_hybrid" variable specifies whether to do a plot for adaptive smoothing method
############################################


import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.tri as tri
import datetime
from matplotlib.colors import ListedColormap
from gd_ex1_new import GradientDescent

### location of minima
xmin1 = [0,0]
xmin2 = [-3,3]


### objective functions
def parabola_sphere(x, args=()):
    x = np.reshape(x, (-1, x[0].shape[-1]))
    f_x = np.where(
        x[:, 0] > 0.0,
        0.01 * np.sum(np.square(x-xmin1), axis=-1),
        -1.0 + 0.01 * np.sum(np.square(x-xmin2), axis=-1),
    )
    return f_x
def parabola_gradient(x, args=()):
    x = np.reshape(x, (-1, x[0].shape[-1]))
    df_x = np.zeros(x.shape)
    df_x[np.where(x[:, 0] > 0.0)] = 0.01 * 2 * (x[np.where(x[:, 0] > 0.0)] - xmin1)
    df_x[np.where(x[:, 0] <= 0.0)] = 0.01 * 2 * (x[np.where(x[:, 0] <= 0.0)] -xmin2)
    return df_x

fmin = parabola_sphere
grad_min = parabola_gradient

### plots the number of function evaluation vs the function value
Plot_landscape_smooth = True ### if set to false, does not plot GSM optimization steps and landscape
Plot_landscape_hybrid = True ### if set to false, does not plot ASM optimization steps and landscape
Plot_landscape_exact = True  ### if set to false, does not plot underlying objective landscape
seperate_landscape = True ### if set to true, will plot the function landscape without optimization iterates
do_work = False
##do_work = True

### specify name of dataset to load
date_time = 'final_exrun_new' ### where the generated data is stored

pathing = './Ex1_' + date_time ### path to dataset
### specifying paths to metadata and landscape paths
metadata = pathing + '/metadata.npy'
filename_ex = pathing + '/Landscape_det.npy'
filename_stoch = pathing + '/Landscape_stoch.npy'
filename_hyb = pathing + '/Landscape_hyb.npy'


### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']

for i in range(0,max_try):
    filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
    filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
        
    f3 = np.load(filename3)
    sol3 = f3['sol3']
    
    
    f1 = np.load(filename1)
    sol1 = f1['sol1']

### meshgrid for landscape
if do_work: 
    resolution = 100
    x_1 = np.linspace(-10, 10, resolution)
    x_2 = np.linspace(-10, 10, resolution)
    x_1, x_2 = np.meshgrid(x_1, x_2, indexing="ij")

### coordinates of starting position
xstart=[9,9]

### colour map for landscape plots
cividis_new = plt.colormaps['viridis']
cmap = ListedColormap(cividis_new(np.linspace(0, 1, 256)))


if Plot_landscape_exact:
    if do_work:
        opt = GradientDescent( xstart = [-1,1], npop = 1, sigma = 1, gradient_available = True,
                   check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,
                   npop_stoch = 1)
        opt.optimize(fmin, iterations= 1, gradient_function = grad_min)
        ii = 0
        z = np.ones((resolution*resolution,1))
        minz = 100
        opt.npop = opt.npop_der
        opt.gradient_available = True
        for entry in np.c_[x_1.ravel(), x_2.ravel()]:
            z[ii] = parabola_sphere([entry])
            ii +=1  
        z = z.reshape(x_1.shape)
        with open(filename_ex,'wb') as f:
                np.savez(f, x_1 = x_1, x_2 = x_2, minz = minz, z = z)
    else:
        data = np.load(filename_ex)
        x_1 = data['x_1']
        x_2 = data['x_2']
        minz = data['minz']
        z = data['z']
    
    fig_ex,ax_ex = plt.subplots(nrows = 1)
    CS = ax_ex.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
    cbar = fig_ex.colorbar(CS)

    ax_ex.axvline(np.sqrt(4), c = 'orange', linestyle = 'dashed')
    ax_ex.axvline(-np.sqrt(4), c = 'orange',linestyle = 'dashed')
    ax_ex.axvline(np.sqrt(2), c = 'red', linestyle = 'dashed')
    ax_ex.axvline(-np.sqrt(2), c = 'red',linestyle = 'dashed')
    ax_ex.axvline(0, c = 'pink', linestyle = 'dashed')
    
    ax_ex.set_xlabel(r'$\mathbf{x}[1]$')
    ax_ex.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)
    fig_ex.suptitle(r'Landscape of $f$')
    if seperate_landscape:
        figplote,axplote = plt.subplots(nrows = 1)
        CS_ls = axplote.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
        cbar2 = figplote.colorbar(CS_ls)
        lens = 0.7
        axplote.axvline(0, c = 'pink', linestyle = 'dashed')
        axplote.axvline(np.sqrt(4), c = 'orange', linestyle = 'dashed')
        axplote.axvline(-np.sqrt(4), c = 'orange',linestyle = 'dashed')
        axplote.axvline(np.sqrt(2), c = 'red', linestyle = 'dashed')
        axplote.axvline(-np.sqrt(2), c = 'red',linestyle = 'dashed')
        points_1 = [-8, -6, -4, -1.6, -1, 0, 1, 1.6, 4, 6, 8]
        points_2 = [4, 3, 2.5, 2, 1, 0, -1, -2, -2.5, -3, -4]
        for k in range(len(points_1)):
                if (points_1[k])**2 < opt.eps_val:
                    z = ((points_1[k])**2-opt.kappa)/(opt.eps_val - opt.kappa)
                    if z>= 0: 
                        leng =  ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                    else:
                        leng = 1
                else:
                    leng = 0
                axplote.scatter(points_1, points_2, color='cyan', s = 4)
                axplote.annotate("", xytext= (points_1[k], points_2[k]), xy=(points_1[k] + leng, points_2[k] + 0), arrowprops=dict(color='cyan', arrowstyle="-", lw = 3))
                axplote.annotate("", xytext= (points_1[k], points_2[k]), xy=(points_1[k] - leng, points_2[k] - 0), arrowprops=dict(color='cyan', arrowstyle="-", lw = 3))
        axplote.set_xlim(-10,10)
        axplote.set_ylim(-10,10)
        axplote.set_xlabel(r'$\mathbf{x}[1]$')
        axplote.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)
        figplote.suptitle(r'Landscape of $f$')
    

if Plot_landscape_smooth:
    ### initialize smoothing function from class
    if do_work:
        opt_st = GradientDescent( xstart = [-1,1], npop = 100, sigma = 1, gradient_available = False)
        opt_st.optimize(fmin, iterations= 1, gradient_function = grad_min)

        ii = 0
        z = np.ones((resolution*resolution,1))
        minz = 100 
        for entry in np.c_[x_1.ravel(), x_2.ravel()]:
            z[ii] = opt_st.compute_val(entry) 
            if z[ii] < minz: ### determine minimum value and position
                minz = z[ii].copy()
                min_coords = entry.copy()
            ii +=1
        z = z.reshape(x_1.shape)
        with open(filename_stoch,'wb') as f:
                np.savez(f, x_1 = x_1, x_2 = x_2, minz = minz, z = z,min_coords= min_coords)
    else:
        data = np.load(filename_stoch)
        x_1 = data['x_1']
        x_2 = data['x_2']
        minz = data['minz']
        z = data['z']
        min_coords = data['min_coords']
    
    fig_full,ax_full = plt.subplots(nrows = 1)
    CS = ax_full.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
    cbar = fig_full.colorbar(CS)
    ax_full.scatter(sol1[:,0], sol1[:,1], c ='aqua', label = 'GSM', marker = 's')
    ax_full.scatter(xstart[0],xstart[1],c = 'magenta', label = 'Starting Point')
    ax_full.scatter(min_coords[0], min_coords[1],c = 'red', label = 'Minimum' )
    ax_full.legend(loc = 'lower left')
    ax_full.set_xlabel(r"$\mathbf{x}[1]$")
    ax_full.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig_full.suptitle(r'Landscape of $F$ and Optimization Steps')


    if seperate_landscape:
        figplot,axplot = plt.subplots(nrows = 1)
        CS_ls = axplot.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
        cbar2 = figplot.colorbar(CS_ls)
        figplot.suptitle(r'Landscape of smooth function $F$')


if Plot_landscape_hybrid:
    ### initialize smoothing function from class 
    if do_work:
        opt_hy = GradientDescent( xstart = [-1,1], npop = 1, sigma = 1, gradient_available = True,
                           check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,
                           npop_stoch = 10)
        opt_hy.optimize(fmin, iterations= 1, gradient_function = grad_min)
        ii = 0
        z = np.ones((resolution*resolution,1))
        minz = 100
        for entry in np.c_[x_1.ravel(), x_2.ravel()]:
            if abs(entry[0])**2 < opt_hy.eps_val:
                opt_hy.npop = opt_hy.npop_stoch
                opt_hy.gradient_available = False
                z[ii] = opt_hy.compute_val(entry)
                opt_hy.npop = opt_hy.npop_der
                opt_hy.gradient_available = True
            else:
                z[ii] = opt_hy.compute_val(entry)
            if z[ii] < minz:
                minz = z[ii].copy()
                min_coords = entry.copy()
            ii +=1  
        z = z.reshape(x_1.shape)
        with open(filename_hyb,'wb') as f:
                np.savez(f, x_1 = x_1, x_2 = x_2, minz = minz, z = z,min_coords= min_coords)
    else: 
        data = np.load(filename_hyb)
        x_1 = data['x_1']
        x_2 = data['x_2']
        minz = data['minz']
        z = data['z']
        min_coords = data['min_coords']
        
    fig_hyb,ax_hyb = plt.subplots(nrows = 1)
    CS = ax_hyb.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
    cbar = fig_hyb.colorbar(CS)

    ax_hyb.axvline(np.sqrt(2), c = 'orange', linestyle = 'dashed')
    ax_hyb.axvline(-np.sqrt(2), c = 'orange',linestyle = 'dashed')

    ind_3 = np.where(np.absolute(sol3[:,0])**2  < 4*10**-0)
    ind_0 = np.setdiff1d(range(len(sol3[:,0])), ind_3)    
    ax_hyb.scatter(sol3[ind_0,0], sol3[ind_0,1], c = 'yellow', label = 'ASM' ,  marker = 'D')
    ax_hyb.scatter(xstart[0],xstart[1],c = 'magenta', label = 'Starting Point')
    ax_hyb.scatter(sol3[ind_3,0], sol3[ind_3,1], c = 'darkorange', label = 'Smoothing is applied', marker = '<', s = 70)
    ax_hyb.scatter(min_coords[0], min_coords[1],c = 'red', label = 'Minimum')
    ax_hyb.legend(loc = 'lower left')
    ax_hyb.set_xlabel(r"$\mathbf{x}[1]$")
    ax_hyb.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig_hyb.suptitle(r'Landscape of $ \mathbb{F}$ and Optimization Steps')
    if seperate_landscape:
        figplot,axplot = plt.subplots(nrows = 1)
        CS_ls = axplot.contourf(x_1, x_2, z, cmap = cmap, levels = 40, zorder = 1)
        cbar2 = figplot.colorbar(CS_ls)
        figplot.suptitle(r'Landscape of hybrid function $\mathbb{F}$')

plt.show()
