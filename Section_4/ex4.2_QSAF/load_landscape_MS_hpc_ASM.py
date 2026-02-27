
import numpy as np
import time
import os
import datetime
from Springclass2D_hpc import SpringFracture
from gd_exMS_hpc import GradientDescent



def run_fun(jj):


    ### specify where file is located
    directory = './LS_ASM/'
##    timecode = 'final_MS'
    timecode = 'test' ### set to test so nothing is accidentally overwritten
    code = timecode + str(jj)
##    suffix = '_04_02_32x32_MS_25_E_575_x2=175_sq' ### suffix of file if you wish to
    suffix = ''
    filename_ls = directory + code + '_Adap_Landscape' + suffix + '.npy'




    method = 'one step'
    L = 21
    B = 41

    pref_dist = 1

    drawdist = 2.5*1.2*pref_dist - pref_dist
    #### if leftupper etc/ no spatial dependency
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

            problem.id = '/tests/' + 'dummy' ### dummy id
            problem.standard_str = 1.2
            problem.diss_type = 'L2'   ### dissipation type
            problem.nu = 0.001
            problem.min_eig = 0
            problem.num_eigs = 2
            problem.inhom_material = True ### yes we have design dependency
            problem.save_output = False ### this would cause a lot of data bloat
            problem.print_progress = False ### would flood your console probably

            if len(args)>=1:
                problem.compute_mat_der = True
            else:
                problem.compute_mat_der = False
                
            problem.pref_dist = pref_dist
            problem.alpha_fix_list = [0.5,15,7.5,7.5]

            problem.boundary = ['left_upper','left_lower','upper','lower'] ### boundary conditions
            problem.alpha_list = [entry[0],entry[1]] ### design variables
            
            problem.loading = [[lu_0,lu_1], [ll_0,ll_1],[ll_0,ll_1],[lu_0,lu_1]] ### all left right upper lower boudnary
            problem.dim = [[0,1],[0,1],[0,1],[0,1]]

            problem.loading_spatial_dep = True
            problem.track_eigs = False
            problem.solve(method = method)
            vals.append(
                         0.5*np.abs(problem.energies[0]- problem.energies[-1] + goal_energy)**2
                        + 10*(entry[0] - 1.75)**2
                        )
            ### compute derivative with sensitvities
            if len(args)>=1:
                derivs.append(
                              (problem.energies[0] - problem.energies[-1] + goal_energy) * (problem.energies_mat[0] - problem.energies_mat[-1])
                              +np.array([ 10 * 2 * (entry[0] - 1.75), 0])
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
    
    fmin = Spring_fct ### objective
    fminE = Eig_fct ### levelset function

    sigma = 0.15*1.0 ### size of variance for normal distribution area for sampling
    eps_val = 0.4 
    kappa = 0.2

    aleph = 0.3


    npop_stoch_num = 10 ### number of samples for smoothing

    ### initializing     
    opt3 = GradientDescent(
            xstart=[1.49611468, 0.81455317], npop=1, sigma=sigma, gradient_available = False, 
            check_for_stability = True, use_one_directional_smoothing = True,
            npop_der = 1,npop_stoch = npop_stoch_num, ident = jj
        )
            
    ### start assembly 

    ### performing all the setup steps so we dont have to do a faux optimization run (would take long)
    opt3.unchanged_mu = False
    opt3.eps_val = eps_val**2 ### this is the true epsilon value, the square of the handed over value, same for kappa
    opt3.kappa = kappa**2  
    opt3.Armijo = False
    opt3.aleph = aleph

    ### set up functions so it can evaluate what it has to
    opt3.obj = lambda x: fmin(x)
    opt3.obj_grad = lambda x: fmin(x, ['derivative'])
    opt3.obj_eigs = lambda x: fminE(x, ['eigs', 'derivative'])
    
    opt3.generate_samples() ### generate samples and weights for kernel
    opt3.generate_weights()


    vals_adap = []


    num_samples = 32 ### 32x32 grid


    ### start assembling landscape
    
    add = 0

    xx = add + np.linspace(0.1,2,num_samples)
    yy = xx + 0.001 

    x_1, x_2 = np.meshgrid(xx, yy, indexing="ij")

    for i in range( jj*int(num_samples**2/64), (jj+1)*int(num_samples**2/64)):

        print('starting iteration ',i)

        start = time.time()
        ff_E = fminE([[x_1.ravel()[i],x_2.ravel()[i]]])
        if ff_E**2 < eps_val**2: 
            opt3.gradient_available = False
            opt3.npop = npop_stoch_num
            ff_ad = opt3.compute_val(np.asarray([x_1.ravel()[i],x_2.ravel()[i]]))

        else:
            opt3.gradient_available = True
            opt3.npop = 1
            ff_ad = fmin([[x_1.ravel()[i],x_2.ravel()[i]]])
            direction = np.zeros((2,))
        vals_adap.append(ff_ad)


        end = time.time()
        length = end - start

        ### in case you want to know what count youre on and how long one simulation is taking
##            print("It took", length, "seconds!")
##            print('done iteration ',i)


    vals_adap = np.asarray(vals_adap)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename_ls, 'wb') as f:
        np.savez(f, vals_adap = vals_adap, x_1 = x_1, x_2 = x_2, eps_val = eps_val, kappa = kappa) ### adaptive

