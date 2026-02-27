import numpy as np
import os as os
import datetime
import time
from numpy import exp
from scipy.sparse.linalg import eigs
from Springclass2D_hpc_backstepping import SpringFracture

def run_fun(jj):

    typing = 6

    method = 'one step'
    diss_types = ['L2','KV'] ### type of norm penalty
    max_steps = 100 ### maximum number of timesteps
    Ntnn = True ### whether to use ntnn or just nn
    inhom_mat = True
    print_progress = True
    save_output = True
    loading_spatial = True
    nulist = [0.001] ### penalty parameter size
    
    chosen_diss = diss_types[0]

    ### deisgns to evaluate
    x_1 = np.linspace(1,4,50)
    x_2 = np.linspace(0.5,3.5,50) 
    
##    timecode = 'EX_post_nu_0001_pull25_o15_f75_s75_nebendiag_ex_' ### prefix for foldernames of where simulation data will be stored
    timecode = 'test' ### set to test so nothing is overwritten by accident
    code = timecode

    
    directory = './QSAF_cracks/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range( len(x_1)):


        print('starting iteration ',i)
        
        L = 11
        B = 21
        
        problem = SpringFracture(L,B,max_steps, ntnn = Ntnn, ref_fact = 5)### ref fact determines the refinement factor at time of fracture (time is refined by factor 5 for a while until fracturing over)
        
        problem.id = directory + code + 'alph0_%f'%(x_1[i]) + '_alph1_%f'%(x_2[i]) ### id of simulation, will save data to this folder
        problem.standard_str = 1.2
        problem.diss_type = chosen_diss   ### dissipation type
        problem.nu = nulist[0]
        problem.min_eig = 0
        problem.num_eigs = 4 ### number of smallest eigenvalues to determine
        problem.inhom_material = inhom_mat
        problem.save_output = True ### saves the output
        problem.print_progress = print_progress ### print progress of simulation (will print out a bunch of info in each timestep, if you dont want this set to False)
        
        problem.pref_dist = 1
        problem.alpha_list = [x_1[i],x_2[i]]
        
        problem.alpha_fix_list = [0.5,15,7.5,7.5] ### fixed material strength parameters

        problem.boundary = ['left_upper','left_lower','upper','lower'] ### bc codes


        drawdist = 2.5*1.2*problem.pref_dist - problem.pref_dist
        
        if loading_spatial:
            #### if leftupper etc/ no spatial dependency
            lu_0 = lambda x,y: 0
            lu_1 = lambda x,y: -(1 - y[::2]/problem.B) * drawdist * x
            lu_1 = lambda x,y: -(1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x
            ll_0 = lambda x,y: 0
            ll_1 = lambda x,y: (1 - y[::2]/problem.B) * drawdist * x
            ll_1 = lambda x,y: (1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x

            problem.loading = [[lu_0,lu_1], [ll_0,ll_1],[ll_0,ll_1],[lu_0,lu_1]] ### all left right upper lower boudnary no rs
            problem.dim = [[0,1],[0,1],[0,1],[0,1]] ### no rs

        problem.loading_spatial_dep = loading_spatial ### if set to False, then assumes that boundary conditions are not dependent on space but only on time

        
        problem.solve(method = method)


    
