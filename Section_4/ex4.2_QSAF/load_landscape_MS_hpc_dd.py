import numpy as np
import os as os
import datetime
import time
from numpy import exp
from scipy.sparse.linalg import eigs

from Springclass2D_hpc_backstepping import SpringFracture


def run_fun(jj):

    method = 'one step'
    diss_types = ['L2','KV'] ### type of norm penalty
    max_steps = 100 ### maximum number of timesteps
    Ntnn = True ### whether to use ntnn or just nn
    inhom_mat = True
    print_progress = True ### if you dont want progress of each simulation printed for you, set False
    save_output = False ### fi you want data stored for each simulation in a folder, set True
    loading_spatial = True
    nulist = [0.001] ### penalty parameter size
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
    eigs_list_2 = []
    eigs_list_extra = []

    chosen_diss = diss_types[0] ##L2 dissipation

    num_samples = 16  # or keep your 32 if preferred

    bot_val= 0.05  # or keep your 32 if preferred
    top_val= 4  # or keep your 32 if preferred
    
    xx = np.linspace(bot_val, top_val, num_samples)
    yy = np.linspace(bot_val, top_val, num_samples)
    X, Y = np.meshgrid(xx, yy, indexing="ij")

    # Define a diagonal refinement "belt"
    width = 0.3  # controls the thickness of the diagonal region
    mask = np.abs(X - Y) < width  # region near the diagonal

    # Base grid (coarse)
    x_1 = X.copy()
    x_2 = Y.copy()

    # Refine points within the belt (denser sampling)
    refinement_factor = 3  # how much finer the diagonal region is
    x_ref = np.linspace(bot_val, top_val, num_samples * refinement_factor)
    y_ref = np.linspace(bot_val, top_val, num_samples * refinement_factor)
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing="ij")

    mask_ref = np.abs(X_ref - Y_ref) < width  # refined belt region

    # Combine the coarse and refined points
    x_1 = np.concatenate([X[~mask].ravel(), X_ref[mask_ref].ravel()])
    x_2 = np.concatenate([Y[~mask].ravel(), Y_ref[mask_ref].ravel()])


##    timecode = 'LS_32x32_post_nu_0001_dense_diag_4x4_pull25_o15_f75_s75'
##    timecode = 'LS_32x32_post_nu_0001_dense_diag_2x2_pull25_o15_f75_s75'
    timecode = 'test' ### so nothing is accidentally overwritten
    
    code = timecode + str(jj)
    directory = './LS_f/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename_ls = './LS_f/' + code + '_Spring_Landscape.npy' ### directory where data is stored at the end
    

    lens = int(len(x_1)/32)+1 ### length of vector divided by number of processes used

    for i in range( jj*lens, min( (jj+1)*lens, len(x_1))):


        print('starting iteration ',i)
        
        L = 21
        B = 41
        
        problem = SpringFracture(L,B,max_steps, ntnn = Ntnn, ref_fact = 5) ### ref fact controls the zoom factor at time of fracture
        problem.id =  code + 'alph0_%f'%(x_1.ravel()[i]) + '_alph1_%f'%(x_2.ravel()[i])
        
        problem.standard_str = 1.2
        problem.diss_type = chosen_diss   ### dissipation type
        problem.nu = nulist[0]
        problem.min_eig = 0
        problem.num_eigs = 4
        problem.inhom_material = inhom_mat

        
        problem.print_progress = print_progress
        
        problem.pref_dist = 1
        problem.alpha_list = [x_1.ravel()[i],x_2.ravel()[i]]
        problem.alpha_fix_list = [0.5,15,7.5,7.5]
        
        problem.boundary = ['left_upper','left_lower','upper','lower'] ### ls no code no rs
        drawdist = 2.5*1.2*problem.pref_dist - problem.pref_dist

        lu_0 = lambda x,y: 0
        lu_1 = lambda x,y: -(1 - y[::2]/problem.B) * drawdist * x
        lu_1 = lambda x,y: -(1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x
        ll_0 = lambda x,y: 0
        ll_1 = lambda x,y: (1 - y[::2]/problem.B) * drawdist * x
        ll_1 = lambda x,y: (1 - np.sin(y[::2]/problem.B *np.pi/2)) * drawdist * x

        
        problem.loading = [[lu_0,lu_1], [ll_0,ll_1],[ll_0,ll_1],[lu_0,lu_1]] ### all left right upper lower boudnary no rs
        problem.dim = [[0,1],[0,1],[0,1],[0,1]] ### no rs

        problem.loading_spatial_dep = loading_spatial
        
        problem.track_eigs = True
        
        problem.solve(method = method)
        
        energy = problem.energies[-1]
        energy_og = problem.energies[0]
        energy_mat = problem.energies_mat[-1]
        energy_og_mat = problem.energies_mat[0]

        #### prints out the energy value for each design, uncomment if you want that
##        print('energy final:', energy)
##        
##        print('energy difference:', energy_og - energy)
##
##        print('energy derivative:', energy_og_mat-energy_mat)


        ####### make lists ##############
        energy_list.append(energy)
        energy_og_list.append(energy_og)
        energy_diff_list.append(energy_og-energy)
        energy_mat_0_list.append(energy_mat[0])
        energy_mat_1_list.append(energy_mat[1])
        energy_og_mat_0_list.append(energy_og_mat[0])
        energy_og_mat_1_list.append(energy_og_mat[1])
        energy_diff_mat_0_list.append(energy_og_mat[0] - energy_mat[0])
        energy_diff_mat_1_list.append(energy_og_mat[1] - energy_mat[1])
        eigs_list.append(np.min(problem.eigsn, axis = 1))
        eigs_list_2.append(np.min(problem.eigsn[:,1:], axis = 1)) ### useless 
        eigs_list_extra.append(problem.eigsn)

    with open(filename_ls, 'wb') as f:
        np.savez(f, eigs = eigs_list, eigs_2 = eigs_list_2, eigs_extra = eigs_list_extra, 
                 energy = energy_list, energy_og = energy_og_list, energy_diff = energy_diff_list,
                 energy_mat_0 = energy_mat_0_list, energy_mat_1 = energy_mat_1_list,
                 energy_og_mat_0 = energy_og_mat_0_list, energy_og_mat_1 = energy_og_mat_1_list,
                 energy_diff_mat_0 = energy_diff_mat_0_list, energy_diff_mat_1 = energy_diff_mat_1_list,
                 x_1 = x_1, x_2 = x_2)

    
