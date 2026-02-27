#### plots simulations

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.tri as tri
import os
import datetime
import scipy.interpolate as sp
from Springclass2D_plot import SpringFracture

method = 'one step'
diss_types = ['L2','KV'] ### type of norm penalty
max_steps = 100 ### maximum number of timesteps
Ntnn = True ### whether to use ntnn or just nn
inhom_mat = True
print_progress = True
save_output = False
loading_spatial = True
nulist = [0.001] ### penalty parameter size

#### deisgns to generate images for
x_1l = [1.051020, 1.173469, 1.908163, 1.969388]
x_2l = [1.551020, 1.673469, 2.408163, 2.469388]
x_1 = np.ones((4,))
x_2 = np.ones((4,))
x_1[:] = x_1l
x_2[:] = x_2l
for i in range(len(x_1)):

    timecode = 'EX_post_nu_0001_pull25_o15_f75_s75_nebendiag_ex_' +'alph0_%f'%(x_1[i]) + '_alph1_%f'%(x_2[i])

    code = timecode
    directory = './2dsprings/QSAF_cracks/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('starting iteration ',i)

    filename_meta = directory + code + '/meta.npy'
    mdata = np.load(filename_meta)

    L = mdata['L']
    B = mdata['B']
    Ntnn = mdata['ntnn']
    max_steps = mdata['max_steps']

    #### this initalizes the construction of the grid, we will not solve anything, just create the same states
    problem = SpringFracture(L,B,max_steps, ntnn = Ntnn)

    problem.id = code 
    
    problem.standard_str = 1.2 ### only set for reference
    problem.diss_type = diss_types[0]   ### dissipation type
    problem.nu = nulist[0]
    problem.inhom_material = inhom_mat ### set true 
    problem.save_output = True
    problem.print_progress = print_progress
    problem.compute_mat_der = False ## set this to true if also want plot sof the design derivative
        
    problem.pref_dist = 1
    problem.alpha_list = [x_1[i],x_2[i]] ### so th emethod knows two design variables are active
    problem.alpha_fix_list = [0.5,15,7.5,7.5] ### doesnt matter for plotting, but needs to be set in same way as simulation so the method knows whats active


    problem.boundary = ['left_upper','left_lower','upper','lower'] ### ls no code

    drawdist = 2.5*1.2*problem.pref_dist - problem.pref_dist ### unfortunately i forgot to save this value, so you have to remember it

    ### boundary condition functions
    lu_0 = lambda x,y: 0
    lu_1 = lambda x,y: -(1 - y[::2]/problem.B) * drawdist * x
    lu_1 = lambda x,y: -(1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x
    ll_0 = lambda x,y: 0
    ll_1 = lambda x,y: (1 - y[::2]/problem.B) * drawdist * x
    ll_1 = lambda x,y: (1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x
    

    problem.loading = [[lu_0,lu_1], [ll_0,ll_1],[ll_0,ll_1],[lu_0,lu_1]] ### all left right upper lower boudnary
    problem.dim = [[0,1],[0,1],[0,1],[0,1]]

    problem.loading_spatial_dep = loading_spatial ### doesnt really matter, set for reference
    problem.folder = './2dsprings/QSAF_cracks/' + str(problem.id) ### manually set reference destination for method to save data to
    
    filename = problem.folder + '/data.npy'
    fdata = np.load(filename)

    ### set all data for method to reference
    problem.Y = fdata['x']
    problem.X = problem.Y[:,0].reshape(L*B,2).copy()
    problem.eigs = fdata['eigs']
    problem.eigsn = fdata['eigsn']
    problem.T = fdata['T']
    print(fdata['eigsn'].shape,fdata['T'].shape)
    problem.hist_list = fdata['r_hist']
    problem.R_1 = fdata['R_1']
    problem.R_2 = fdata['R_2']
    problem.hist_ntnn_list = fdata['r_hist_ntnn']
    problem.R_1_ntnn = fdata['R_1_ntnn']
    problem.R_2_ntnn = fdata['R_2_ntnn']

    problem.time_needed = 0 ### only so the method has a value

    ### plot eigenvalues
    problem.plot_eigsn(print_type = 'pdf')### plots the images to problem.id folder
##    problem.plot_eigsn(print_type = 'pdf', suffix = 'test', zieldirectory = '/test')### alternative, plots images to a specified zieldirectory with a special suffix


    ### plot system evolution of crack
    problem.plot_system(timelist = range(0,len(fdata['T']),int(len(fdata['T'])/3)) , print_int = int(len(fdata['T'])/3), print_type = 'pdf')
##    problem.plot_system(range(0,len(fdata['T']),int(len(fdata['T'])/3)),int(len(fdata['T'])/3), print_type = 'pdf', suffix = 'test', zieldirectory = '/test') ### alternative, plots images to a specified zieldirectory with a special suffix

######
##    plot_system needs:  timelist, a list of entries you want to print the system state at, print_int = interval in wihch the method tells you its printing a timestep


    
