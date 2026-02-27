############################################
#### Postprocesses the data egenrated from your run, evaluates ASM and GSM at each iterate
############################################

import numpy as np
import os
import datetime
from gd_ex_spring_hpc_new import GradientDescent
from Springclass_hpc_AS import FractureProblem
def run_fun(given_index):

    n_time_steps = 101
    method = 'staggered_multi_step'
    Springprob = FractureProblem(n_time_steps, 1)
    

    def Spring_fct(x, args =()):
        ### calls spring simulation for function evaluation from script Springclass.py
        eigs = []
        vals = []
        derivs = []
        for entry in x:
            Springprob.G1 = entry[0]
            Springprob.G2 = entry[1]
            Springprob.solve(method = method)
            ### compute function value with solution from simulation
            vals.append(
                         - Springprob.sol[-2,-1] 
                        + (entry[0] - 0.7)**2
                        + (entry[1] - 0.9)**2
                        )
            ### compute derivative with sensitvities
            if len(args)>=1:
                
                derivs.append(
                              - Springprob.der[-2,:] 
                              +np.array([ 2*(entry[0] - 0.7), 0])
                              +np.array([ 0,2*(entry[1] - 0.9)])
                              )
            if len(args)>=2:
                ### eigenvalue 
                index = np.argmin(Springprob.eigs[0,:])
                eigs.append(Springprob.eigs[0,index]/(Springprob.G1 + Springprob.G2))

        
        if len(args)>=2:
            return np.asarray(vals), np.asarray(derivs),np.asarray(eigs)
        elif len(args)>=1:
            return np.asarray(vals), np.asarray(derivs)
        else: 
            return np.asarray(vals)


    fmin = Spring_fct
    fmincon = lambda x: Spring_fct(x,['Derivative'])

    sigma = 0.15*1.0 ### size of variance for normal distribution area for sampling
    eps_val = np.sqrt(0.7)
    kappa = np.sqrt(0.5)
    aleph = 0.2

##    pathing = './ExAS_final_new'
    pathing = './test' ### set to test so nothing is overwritten by accident
    metadata = pathing + '/metadata.npy'

    ### load metadata
    meta = np.load(metadata)
    max_try = meta['max_try']
    iter_max = meta['iter_max']
    multi = meta['multi']
    
    ### initialiize data storage vectors
    Fsol1_ex_store = np.zeros((iter_max,max_try))
    x1_store = np.zeros(((iter_max,max_try)))
    
    Fsol3_ex_store = np.zeros((iter_max*multi,max_try))
    x3_store = np.zeros(((iter_max*multi,max_try)))


    npop = 100
    npop_stoch = 10
    opt1 = GradientDescent(
        xstart=[0.69, 0.9], npop= npop, sigma=sigma, gradient_available = False, 
        npop_der = 1,npop_stoch = npop_stoch
    )
    opt3 = GradientDescent(
                xstart=[8,8], npop=1, sigma=sigma, gradient_available = False, 
                check_for_stability = True, use_one_directional_smoothing = True,
                npop_der = 1,npop_stoch = npop_stoch
            )
    opt1.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    opt3.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)

    ### define objective functions
    fmin = Spring_fct
    fmincon = lambda x: Spring_fct(x,['Derivative'])


    i = given_index
    print('-------- computing try %i -----------' %i)
    filename2 = pathing + '/Run_' + str(i) + '_results_stoch.npy'
    filename3 = pathing + '/Run_' + str(i) + '_results_hyb.npy'
        
    f3 = np.load(filename3)
    sol3 = f3['sol3']
    x3 = f3['x3']
    x3_store[:,i] = x3.copy()
    
    f1 = np.load(filename2)
    sol1 = f1['sol2']
    x1 = f1['x2']
    x1_store[:,i] = x1.copy()

    ii = 0
    opt3.gradient_available = False
    for entry in sol3:
        ff = opt3.compute_val(entry)
        Fsol3_ex_store[ii,i] = ff
        ii+=1
    ii = 0
    for entry in sol1:
        ff = opt1.compute_val(entry)
        Fsol1_ex_store[ii,i] = ff
        ii+=1
        
    Fsol1_store = Fsol1_ex_store
    Fsol3_store = Fsol3_ex_store
    with open(pathing + '/Run_' + str(i) + 'Vals_stoch.npy', 'wb') as f:
        np.savez(f, Fsol1_store = Fsol1_store, x1_store = x1_store)

    with open(pathing + '/Run_' + str(i) + 'Vals_hybrid.npy', 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store, x3_store = x3_store)


















