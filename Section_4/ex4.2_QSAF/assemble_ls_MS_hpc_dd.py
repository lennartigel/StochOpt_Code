
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.tri as tri
import os
import datetime

method = 'one step'
diss_types = ['L2','KV'] ### type of norm penalty
max_steps = 100 ### maximum number of timesteps
Ntnn = True ### whether to use ntnn or just nn
inhom_mat = True
print_progress = True
save_output = False
nulist = [0.001] ### penalty parameter size


endtime = 1

directory = './LS_f/'

assemble = True ### set to true if you want to assemble a created landscape, false if only plotting, default is false so nothing is overwritten by accident
assemble = False



timecode = 'LS_32x32_post_nu_0001_dense_diag_4x4_pull25_o15_f75_s75_'
##timecode = 'LS_32x32_post_nu_0001_dense_diag_2x2_pull25_o15_f75_s75_'



if assemble:

    energy_list = []
    energy_og_list = []
    energy_mat_0_list = []
    energy_mat_1_list = []
    energy_og_mat_0_list = []
    energy_og_mat_1_list = []
    energy_diff_mat_0_list = []
    energy_diff_mat_1_list = []
    energy_diff_list = []
    eigs_list = []
    eigs1_list = []
    eigs2_list = []
    eigs3_list = []
    
    eigs1_normal_list = []
    eigs0_normal_list = []
    eigs_diag =[]
    x1_diag = []
    x2_diag = []

    lens = 0


    for jj in range(32):
        print('doing number:', jj)
        code = timecode + str(jj)
        filename_ls = './LS_f/' + code + '_Spring_Landscape.npy'
        data = np.load(filename_ls, allow_pickle = True)

        energy_list = np.append(energy_list,data['energy'])
        energy_og_list = np.append(energy_og_list,data['energy_og'])
        energy_diff_list = np.append(energy_diff_list,data['energy_diff'])
        energy_mat_0_list = np.append(energy_mat_0_list,data['energy_mat_0'])
        energy_mat_1_list = np.append(energy_mat_1_list,data['energy_mat_1'])
        energy_og_mat_0_list = np.append(energy_og_mat_0_list,data['energy_og_mat_0'])
        energy_og_mat_1_list = np.append(energy_og_mat_1_list,data['energy_og_mat_1'])
        energy_diff_mat_0_list = np.append(energy_diff_mat_0_list,data['energy_diff_mat_0'])
        energy_diff_mat_1_list = np.append(energy_diff_mat_1_list,data['energy_diff_mat_1'])
        eigs_list = np.append(eigs_list,data['eigs'][:,0])
        
        indeces = np.argmin(data['eigs_extra'][:,0,:],axis = 1)
        ddata = np.zeros(data['eigs_extra'][:,1,1].shape)
        ddata2 = np.zeros(data['eigs_extra'][:,1,1].shape)
        ddata3 = np.zeros(data['eigs_extra'][:,1,1].shape)
        ddata_normal = np.zeros(data['eigs_extra'][:,1,1].shape)
        ddata_normal0 = np.zeros(data['eigs_extra'][:,0,1].shape)

        
        for kk in range(len(indeces)):
            minind = np.argmin(data['eigs_extra'][kk,0,:])
            minind1 = np.argmin(data['eigs_extra'][kk,1,:])
            minind2 = np.argmin(data['eigs_extra'][kk,1,minind1:])

            
            ddata[kk] = data['eigs_extra'][kk,1,minind]
            ddata2[kk] = data['eigs_extra'][kk,2,minind]
            ddata3[kk] = data['eigs_extra'][kk,3,minind]
            ddata_normal[kk] = data['eigs_extra'][kk,1,minind]/(data['x_1'][jj*lens +kk] + data['x_2'][jj*lens+kk] +0.1)
            ddata_normal0[kk] = data['eigs_extra'][kk,0,minind]/(data['x_1'][jj*lens +kk] + data['x_2'][jj*lens+kk] +0.1)

        lens = len(data['energy'])
    
        eigs1_list = np.append(eigs1_list,ddata) ### lambda1 values at minimum of lambda0
        eigs2_list = np.append(eigs2_list,ddata2) ### lambda2 values at minimum of lambda0
        eigs3_list = np.append(eigs3_list,ddata3) ### lambda3 values at minimum of lambda0
        
        eigs1_normal_list = np.append(eigs1_normal_list,ddata_normal) ### normalized lambda1
        eigs0_normal_list = np.append(eigs0_normal_list,ddata_normal0) #### normalized lambda0
        
        num_samples = 32
        ind = jj
        
    code = timecode + str(0)
    filename_ls = './LS_f/' + code + '_Spring_Landscape.npy'
    data = np.load(filename_ls, allow_pickle = True)

    x_1 = data['x_1']
    x_2 = data['x_2']

    filename_ls_full = './LS_f/' + code + '_Spring_Landscape.npy'
    with open(filename_ls_full, 'wb') as f:
##        np.savez(f, samples = samples, x_1 = x_1, x_2 = x_2, xx_1 = xx_1, xx_2 = xx_2, m = m, grad = grad, dff = dff, line = line, ff = ff, xx_norm = xx_norm, c = c)
        np.savez(f, eigs = eigs_list, eigs1 = eigs1_list, eigs2 = eigs2_list, eigs3 = eigs3_list,
                 eigs1_normal = eigs1_normal_list, eigs0_normal = eigs0_normal_list,
                 energy = energy_list, energy_og = energy_og_list, energy_diff = energy_diff_list,
                 energy_mat_0 = energy_mat_0_list, energy_mat_1 = energy_mat_1_list,
                 energy_og_mat_0 = energy_og_mat_0_list, energy_og_mat_1 = energy_og_mat_1_list,
                 energy_diff_mat_0 = energy_diff_mat_0_list, energy_diff_mat_1 = energy_diff_mat_1_list,
                 x_1 = x_1, x_2 = x_2)

else:
    filename_ls_full = './LS_f/'  + timecode + '_Spring_Landscape.npy'
    data = np.load(filename_ls_full, allow_pickle = True)
    x_1 = data['x_1']
    x_2 = data['x_2']
    
    energy = data['energy']
    energy_og = data['energy_og']
    energy_diff = data['energy_diff']
    energy_mat_0 = data['energy_mat_0']
    energy_mat_1 = data['energy_mat_1']
    energy_og_mat_0 = data['energy_og_mat_0']
    energy_og_mat_1 = data['energy_og_mat_1']
    energy_diff_mat_0 = data['energy_diff_mat_0']
    energy_diff_mat_1 = data['energy_diff_mat_1']

    eigs = data['eigs']
    eigs1 = data['eigs1']
    eigs2 = data['eigs2']
    eigs3 = data['eigs3']
    eigs1_normal = data['eigs1_normal']
    eigs0_normal = data['eigs0_normal']


    cividis_new = plt.colormaps['viridis']
    cmap = ListedColormap(cividis_new(np.linspace(0, 1, 256)))


    fig2,ax2 = plt.subplots(nrows = 1)
    CS2 = ax2.tricontourf(x_1, x_2, np.asarray(eigs), cmap = cmap, levels = 40, zorder = 1)
    cbar = fig2.colorbar(CS2)
    ax2.set_xlabel(r"$\mathbf{x}[1]$")
    ax2.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig2.suptitle(r'Landscape of eigs')
    ax2.yaxis.set_label_coords(-0.12,0.53)
    
#### 
    fig2b,ax2b = plt.subplots(nrows = 1)
    CS2 = ax2b.tricontourf(x_1, x_2, np.asarray(eigs1) , cmap = cmap, levels = 40, zorder = 1)
    cbar = fig2b.colorbar(CS2)
    ax2b.set_xlabel(r"$\mathbf{x}[1]$")
    ax2b.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig2b.suptitle(r'Landscape of eigs1')
    ax2b.yaxis.set_label_coords(-0.12,0.53)

    plotval = np.where((np.asarray(eigs1)) < 0.4,np.asarray(eigs1),0.4)
    plotval = np.where((np.asarray(eigs1)) > -0.4, np.asarray(eigs1),-0.4)
    fig2bb,ax2bb = plt.subplots(nrows = 1)
    CS2 = ax2bb.tricontourf(x_1, x_2, plotval , cmap = cmap, levels = 40, zorder = 1)
    cbar = fig2bb.colorbar(CS2)
    ax2bb.set_xlabel(r"$\mathbf{x}[1]$")
    ax2bb.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    fig2bb.suptitle(r'Landscape of eigs1 with cutoff noise')
    ax2bb.yaxis.set_label_coords(-0.12,0.53)
    
####  an assortment of different stuff you can plot if you want
##      ### lambda2 values
##    fig2b2,ax2b2 = plt.subplots(nrows = 1)
##    levels = np.linspace(0,0.45,40)
##    CS2 = ax2b2.tricontourf(x_1, x_2, np.asarray(eigs2), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig2b2.colorbar(CS2)
##    ax2b2.set_xlabel(r"$x_1$")
##    ax2b2.set_ylabel(r"$x_2$")
##    fig2b2.suptitle(r'Landscape of eigs2')
    
##      #### lambda3 values 
##    fig2b3,ax2b3 = plt.subplots(nrows = 1)
##    levels = np.linspace(0,0.45,40)
##    CS2 = ax2b3.tricontourf(x_1, x_2, np.asarray(eigs3), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig2b3.colorbar(CS2)
##    ax2b3.set_xlabel(r"$x_1$")
##    ax2b3.set_ylabel(r"$x_2$")
##    fig2b3.suptitle(r'Landscape of eigs3')
    
##      ### lambda1 normalized
##    fig2c,ax2c = plt.subplots(nrows = 1)
##    CS2 = ax2c.tricontourf(x_1, x_2, np.asarray(eigs1_normal), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig2c.colorbar(CS2)
##    ax2c.set_xlabel(r"$x_1$")
##    ax2c.set_ylabel(r"$x_2$")
##    fig2c.suptitle(r'Landscape of eigs1 normed')
    
##      ### lambda0 normalized
##    fig2c2,ax2c2 = plt.subplots(nrows = 1)
##    CS2 = ax2c2.tricontourf(x_1, x_2, np.asarray(eigs0_normal) + np.asarray(eigs1_normal), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig2c2.colorbar(CS2)
##    ax2c2.set_xlabel(r"$x_1$")
##    ax2c2.set_ylabel(r"$x_2$")
##    fig2c2.suptitle(r'Landscape of eigs0 normed') 
######################################


###### in case you want to plot energy at end time and energy at starting time seperate
##    levels = np.linspace(-3836, -3765,40)
##    fig3,ax3 = plt.subplots(nrows = 1)           
##    CS2 = ax3.tricontourf(x_1, x_2, np.asarray(energy), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig3.colorbar(CS2)
##    ax3.set_xlabel(r"$\mathbf{x}[1]$")
##    ax3.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
##    ax3.yaxis.set_label_coords(-0.12,0.53)
##    fig3.suptitle(r'Landscape of energy end')
##
##    fig3b,ax3b = plt.subplots(nrows = 1)           
##    CS2 = ax3b.tricontourf(x_1, x_2, np.asarray(energy_og), cmap = cmap, levels = 40, zorder = 1)
##    cbar = fig3b.colorbar(CS2)
##    ax3b.set_xlabel(r"$\mathbf{x}[1]$")
##    ax3b.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
##    ax3b.yaxis.set_label_coords(-0.12,0.53)
##    fig3b.suptitle(r'Landscape of energy start')

    
    fig3c,ax3c = plt.subplots(nrows = 1)           
    CS2 = ax3c.tricontourf(x_1, x_2, np.asarray(energy_diff), cmap = cmap, levels = 40, zorder = 1)
    cbar = fig3c.colorbar(CS2)
    ax3c.set_xlabel(r"$\mathbf{x}[1]$")
    ax3c.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    ax3c.yaxis.set_label_coords(-0.12,0.53)
    fig3c.suptitle(r'Landscape of energy diff')


    fig3c4,ax3c4 = plt.subplots(nrows = 1)
    levels = np.linspace(0,20, 40)
    CS2 = ax3c4.tricontourf(x_1, x_2, 0.5*np.abs(np.asarray(energy_diff +57.5))**2 + 10*(x_1-1.75)**2 , cmap = cmap, levels = 40, zorder = 1)
    cbar = fig3c4.colorbar(CS2)
    levels = [0.2**2, 0.4**2]
    CS_boundaries = ax3c4.tricontour(
        x_1, x_2, ((x_1-x_2)**2),
        levels=levels,
        colors=['red','orange'],
        linestyles=['--', '--'],
        linewidths=1,
        zorder=3
    )
    ax3c4.set_xlabel(r"$\mathbf{x}[1]$")
    ax3c4.set_ylabel(r"$\mathbf{x}[2]$", rotation = 0)
    ax3c4.yaxis.set_label_coords(-0.12,0.53)
    ax3c4.set_ylim(np.min(x_2), np.max(x_2))
    ax3c4.set_xlim(np.min(x_1), np.max(x_1))
    fig3c4.suptitle(r'Landscape of chosen objective $f$')


    
    plt.show()





