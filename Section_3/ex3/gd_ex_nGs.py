####script to optimize a design optimization problem with a spring system subproblem.
####the spring subproblem is solved via a simulation method implemented in Springclass, which
####    also computes the necessary sensitivities for the derivative of the design objectove function.

import numpy as np
import os
import datetime
from scipy.integrate import quad, dblquad
from Gradient_Descent_variable_circle import GradientDescent


def Spring_fct(x,args = ()):
    eigs = []
    eigs_der = []
    vals = []
    derivs = []

    evalue = 5

    push = 0
    push2 = 0

    ### discontinuous parabola, with instant drop at x= 0
    for entry in x:
        if entry[0]-entry[1] +push >= 0 and entry[1]+entry[0] +push2>= 0:
            vals.append( np.abs(entry[1])+np.abs(entry[0]) -6*evalue
                    )
        elif entry[0]-entry[1] +push >= 0 and entry[1]+entry[0] +push2< 0:
            vals.append( np.abs(entry[0]+entry[1]) +2*evalue
                    )
        elif entry[0]-entry[1] +push < 0 and entry[1]+entry[0] +push2>= 0:
            vals.append( np.abs(entry[0]-entry[1]) +evalue
                    )
        elif entry[0]-entry[1] +push < 0 and entry[1]+entry[0] +push2< 0:
            vals.append( np.abs(entry[0]) + np.abs(entry[1]) +np.abs(entry[1]+entry[0] - 3) + 2*evalue
                    )
        if len(args)>=1:
            df_x = np.zeros(2)  
            if entry[0]-entry[1] +push> 0 and entry[1]+entry[0] +push2> 0:
                df_x[1] = np.sign(entry[1])
                df_x[0] = np.sign(entry[0])
            elif entry[0]-entry[1] +push> 0 and entry[1]+entry[0] +push2< 0:
                df_x[1] = np.sign(entry[1]+entry[0])
                df_x[0] = np.sign(entry[1]+entry[0])
            elif entry[0]-entry[1] +push< 0 and entry[1]+entry[0] +push2> 0:
                df_x[1] = -np.sign(-entry[1]+entry[0])
                df_x[0] = np.sign(-entry[1]+entry[0])
            elif entry[0]-entry[1] +push<= 0 and entry[1]+entry[0] +push2 <= 0:
                df_x[0] = np.sign(entry[0])+np.sign(entry[1]+entry[0])
                df_x[1] = np.sign(entry[1])+np.sign(entry[1]+entry[0])
            derivs.append(df_x
                )
        if len(args)>=2:
            eigs.append([np.abs(entry[0]-entry[1]) +push,np.abs(entry[0]+entry[1]) +push2])
            eigs_der.append([np.asarray([1,-1]), np.asarray([1,1])])
    if len(args)>=2:
        return np.asarray(vals), np.asarray(derivs),np.asarray(eigs), np.asarray(eigs_der)
    elif len(args)>=1:
        return np.asarray(vals), np.asarray(derivs)
    else: 
        return np.asarray(vals)

### define objective functions
fmin = Spring_fct
fmincon = lambda x: Spring_fct(x,['Derivative'])


### meta data for optimization
iter_max = 100 ### max number of optim steps
multi = 1 ### multiplication factor to balance number of func evals between methods
max_try = 100 ### max number of random runs

### meta data for optimization
##iter_max = 100 
##multi = 1
##max_try = 1


pop_per_dir = 10
sigma = 1 ### size of variance for normal distribution area for sampling

### these are squared in the method itself, list lets you set individual epsilon and kappa for each level set function
### if only one value is provided its set for all level set functions
eps_val = [np.sqrt(9)] 
kappa = [np.sqrt(4)]


aleph = 5 ### alpha_0 for Armijo

## folder name
##date_time = 'final_circ_new' ### path used for 100 runs 
##date_time = 'final_circ_exrun_new' ### path used for example run
date_time = 'test'

pathing = './Ex_nGs_' + date_time
if not os.path.exists(pathing): 
    os.makedirs(pathing)

### meta data file
metadata = pathing + '/metadata.npy'
with open(metadata, 'wb') as f:
        np.savez(f,iter_max = iter_max, multi= multi, max_try = max_try, sigma = sigma)

for i in range(max_try):
    ## folder name
    print('------------------- Iteration %i -------------------'%i)
    filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
    filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
    if not os.path.exists(pathing): 
        os.makedirs(pathing)

    ### starting point (random or deterministic, you choice)
##    xstart=[0,15] ### for example run
    xstart = 20*( np.random.rand(2)*2-1)
    print('############## HYBOPT %i ################' %(i))
    print(xstart)
    

    ### optimization methods
    opt3 = GradientDescent(
        xstart=xstart, npop=1, sigma=sigma, gradient_available = True, 
        check_for_stability = True, use_one_directional_smoothing = True,
        npop_der = 1,npop_stoch = pop_per_dir, num_id_funcs = 2
    )
    

    ### start optimization 
    opt3.optimize(fmin, iterations=iter_max*multi, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    sol3 = opt3.history["solution"]
    sol3 = np.asarray(sol3)

    ### read out results
    x3 = opt3.history['NumIters']

    ### save results to file
    with open(filename3, 'wb') as f:
        np.savez(f, sol3 = sol3, x3 = x3)

    print('############## STOCHOPT %i ################' %(i))
    ### optimization methods
    opt = GradientDescent(
        xstart=xstart, npop=pop_per_dir* pop_per_dir, sigma=sigma, gradient_available = False, 
        check_for_stability = False, use_one_directional_smoothing = False,
        npop_der = 1,npop_stoch = pop_per_dir, num_id_funcs = 0
    )

    
    ### start optimization 
    opt.optimize(fmin, iterations=iter_max, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    sol = opt.history["solution"]
    sol = np.asarray(sol)

    ### read out results
    x = opt.history['NumIters']

    ### save results to file
    with open(filename1, 'wb') as f:
        np.savez(f, sol1 = sol, x1 = x)
