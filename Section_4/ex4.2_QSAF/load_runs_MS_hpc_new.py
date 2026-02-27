############################################
#### Script to load the generated data from your run of the gd_exMS_hpc.py script
#### and postprocess them, so theyre visually more appealing.
############################################

import numpy as np
import os
import datetime
from Springclass2D_hpc import SpringFracture
from gd_exMS_hpc import GradientDescent

def run_fun(given_index):
    
    method = 'one step'
    L = 21
    B = 41

    pref_dist = 1

    drawdist = 2.5*1.2*pref_dist - pref_dist
    
    #### boundary condition functions
    lu_0 = lambda x,y: 0
    lu_1 = lambda x,y: -(1 - y[::2]/B) * drawdist * x
    lu_1 = lambda x,y: -(1 - np.sin(y[::2]/B *np.pi/2)) * drawdist * x
    ll_0 = lambda x,y: 0
    ll_1 = lambda x,y: (1 - y[::2]/B) * drawdist * x
    ll_1 = lambda x,y: (1 - np.sin(y[::2]/B *np.pi/2)) * drawdist * x

    
    goal_energy = 57.5
    def Spring_fct(x, args =()):
        ### calls spring simulation for function evaluation from script Springclass.py
        eigs = []
        vals = []
        derivs = []
        for entry in x:
            problem = SpringFracture(L,B,100, ntnn = True)

            problem.id = '/tests/' + 'dummy' ### dummy directory
            problem.standard_str = 1.2
            problem.diss_type = 'L2'   ### dissipation type
            problem.nu = 0.001
            problem.min_eig = 0
            problem.num_eigs = 2
            problem.inhom_material = True ### true we have design dependency
            problem.save_output = False ### no would bloat folders 
            problem.print_progress = False ### no would bloat the console with printing

            if len(args)>=1:
                problem.compute_mat_der = True
            else:
                problem.compute_mat_der = False
                
            problem.pref_dist = pref_dist
            problem.alpha_fix_list = [0.5,15,7.5,7.5]

            problem.boundary = ['left_upper','left_lower','upper','lower'] ### ls no code
            problem.alpha_list = [entry[0],entry[1]] ### design variables

            ### boundary conditions specified
            problem.loading = [[lu_0,lu_1], [ll_0,ll_1],[ll_0,ll_1],[lu_0,lu_1]] ### all left right upper lower boudnary
            problem.dim = [[0,1],[0,1],[0,1],[0,1]]

            problem.loading_spatial_dep = True
            problem.track_eigs = False
            problem.solve(method = method)
            ### compute function value with solution from simulation
            vals.append(
                         0.5*np.abs(problem.energies[0]- problem.energies[-1] + goal_energy)**2
                        + 10* (entry[0] - 1.75)**2
                        )
            ### compute derivative with sensitvities
            if len(args)>=1:
                derivs.append(
                              (problem.energies[0] - problem.energies[-1] + goal_energy) * (problem.energies_mat[0] - problem.energies_mat[-1])

                              +np.array([ 10 * 2* (entry[0] - 1.75), 0])
                              )
        if len(args)>=1:
            return np.asarray(vals), np.asarray(derivs)
        else: 
            return np.asarray(vals)

    offset = 2
    def Eig_fct(x, args =()):
        ### calls spring simulation for function evaluation from script Springclass.py
        eigs = []
        eigs_der = []
        for entry in x:
            ### compute function value with solution from simulation
            eigs.append( (entry[0]) - (entry[1])
                        )
            ### compute derivative with sensitvities
            if len(args)>=1:
                ### eigenvalue 
                eigs_der.append( np.asarray([1, -1])
                    )
        if len(args) >=1:
            return np.asarray(eigs), np.asarray(eigs_der)
        else:
            return np.asarray(eigs)
                 
    ### define objective functions
    fmin = Spring_fct
    fminE = Eig_fct#
    
    ### meta data for optimization
    iter_max = 30 ### max number of optim steps
    multi = 10 ### multiplication factor to balance number of func evals between methods
    max_try = 32*4 ### max number of random runs
    
    sigma = 0.15*1.0 ### size of variance for normal distribution area for sampling
    eps_val = 0.4
    kappa = 0.2

    aleph = 0.3

    ### create folder structure for saving of data 
##    pathing = './Ex_QSAF_final/'
    pathing = './test' ### set to test so nothing is accidentally overwritten
    i = given_index  
    
    if not os.path.exists(pathing): 
            os.makedirs(pathing)
            
        
    ### meta data file
    metadata = pathing + '/metadata.npy'
    with open(metadata, 'wb') as f:
            np.savez(f,iter_max = iter_max, multi= multi, max_try = max_try, sigma = sigma)


    i = given_index   

    ### load metadata
    iter_max = 30 ### max number of optim steps
    multi = 10 ### multiplication factor to balance number of func evals between methods
    max_try = 32*4 ### max number of random runs
    
    ### initialiize data storage vectors
    Fsol3_ex_store = np.zeros((iter_max*multi,max_try))
    x3_store = np.zeros(((iter_max*multi,max_try)))

    npop_stoch_num = 10 ### number of samples for smoothing

    ### initialize ASM so we can use it to evaluate
    opt3 = GradientDescent(
            xstart=[1.49611468, 0.81455317], npop=1, sigma=sigma, gradient_available = False, 
            check_for_stability = True, use_one_directional_smoothing = True,
            npop_der = 1,npop_stoch = npop_stoch_num, ident = i
        )
    
    ### starting computations
    opt3.unchanged_mu = False ### has no effect

    #### initalize ASM parameters manually
    opt3.eps_val = eps_val**2 ### this is the true epsilon and kappa value, its squared inside the GradientDescent method before optimization
    opt3.kappa = kappa**2 
    opt3.Armijo = False
    opt3.aleph = aleph ### doesnt matter, just for completeness

    ### so the class knows what functions to smooth with
    opt3.obj = lambda x: fmin(x)
    opt3.obj_grad = lambda x: fmin(x, ['derivative'])
    opt3.obj_eigs = lambda x: fminE(x, ['eigs', 'derivative'])

    ### generate samples and weights for kernel
    opt3.generate_samples()
    opt3.generate_weights()

    
    i = given_index
    print('-------- computing try %i -----------' %i)
    filename3 = pathing + '/Run_' + str(i) + '_results_hyb.npy'
        
    f3 = np.load(filename3)
    sol3 = f3['sol3']
    x3 = f3['x3']
    x3_store[:,i] = x3.copy()

    ii = 0
    opt3.gradient_available = False

    ### this is done in case the computation takes too long and times out the hpc time limit
    ### we save in time intervals and can start from last savepoint in this way
    advance = 1
    start_time = datetime.datetime.now()
    datapath = pathing + '/Run_' + str(i) + '_temp_Vals_hybrid.npy'
    ###
    #### load data from last savepoint if exists
    if os.path.exists( datapath): 
        datapathf = np.load(datapath, allow_pickle=True)
        ii = datapathf['current_it']
        Fsol3_ex_store = datapathf['store']

    ### keep computing
    for entry in sol3[ii:]:
        print('-------- computing run %i of try %i -----------' %(ii,i))
        iter_time = datetime.datetime.now() - start_time

        ### if over time interval length, make savepoint
        if iter_time.seconds/(3600) >= 6*advance: 
                advance +=1
                with open(datapath, 'wb') as f:
                    np.savez(f, store = Fsol3_ex_store, current_it = ii)

        if np.linalg.norm(entry-sol3[-1]) > 10**-8:
            ee = Eig_fct([entry])
            if ee**2 < eps_val**2:
##                print('iteration %i of run %i doing smoothing '%(ii,i) ) ### in case you want to know when smoothing is done and when not
                opt3.gradient_available = False ### do smoothing
                opt3.npop = npop_stoch_num
                ff = opt3.compute_val(np.asarray(entry))
            else:
##                print('iteration %i of run %i doing no smoothing '%(ii,i) )
                opt3.gradient_available = True ### dont do smoothing
                opt3.npop = 1
                ff = Spring_fct([entry])
        else:
            ff = Fsol3_ex_store[ii-1,i] ### if we are converged, dont need to keep computing
        Fsol3_ex_store[ii,i] = ff
        ii+=1
    
    Fsol3_store = Fsol3_ex_store

    with open(pathing + '/Run_' + str(i) + 'Vals_hybrid.npy', 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store, x3_store = x3_store)


















