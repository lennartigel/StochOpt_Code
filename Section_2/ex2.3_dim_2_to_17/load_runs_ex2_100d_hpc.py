############################################
#### Script to assemble the generated data from your run and evaluate 
############################################

import numpy as np
import os
import datetime
import scipy.interpolate as sp
from gd_ex2_100d_hpc import GradientDescent

dim_list = range(2,18)
def run_fun(ndim):
##if do_work:
    k = 0
    pathing = './StochOpts/comp_small_new/final_2_to_17_new_' + str(ndim)

    metadata = pathing + '/metadata.npy'

    meta = np.load(metadata)
    max_try = meta['max_try']
    iter_max = meta['iter_max']
    multi = meta['multi']

##    for ndim in dim_list:

    xmin1 = np.zeros((ndim,))

    def abs_sphere(x, args=()):
        ### discontinuous absolute value function, with instant drop at x= 0
        x = np.reshape(x, (-1, x[0].shape[-1]))
        f_x = np.where(
            x[:, 0] > 0.0,
            1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
            -1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
        )
    ##        print(np.sum((x - xmin1) ), x)
        return f_x

    def abs_gradient(x, args=()):
        ### gradient function for discontinuous absolute value function
        x = np.reshape(x, (-1, x[0].shape[-1]))
        df_x = np.zeros(x.shape)        
        df_x[np.where(x[:, 0] > 0.0),0] = 0.1 
        df_x[np.where(x[:, 0] <= 0.0),0] = -0.1 
        df_x[:,1:] = 0.02 * (x-xmin1)[:,1:]
        return df_x


    fmin = abs_sphere
    grad_min = abs_gradient

    ##### ASM 
    Fsol3_store = np.zeros((iter_max*multi,max_try ))
    x3_store = np.zeros(((iter_max*multi,max_try )))

    opt3 = GradientDescent( xstart = np.ones((ndim,)), npop = 1, sigma = 1, gradient_available = True,
                       check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,
                       npop_stoch = 15)
    opt3.optimize(fmin, iterations= 1, gradient_function = grad_min)

    #### GSM
    Fsol1_store = np.zeros((iter_max,max_try ))
    x1_store = np.zeros(((iter_max,max_try )))

    npop_per_dir = 3
    opt1 = GradientDescent( xstart = np.ones((ndim,)), npop = npop_per_dir**ndim, sigma = 1, gradient_available = False, npop_stoch = npop_per_dir)
    opt1.optimize(fmin, iterations= 1, gradient_function = grad_min)

    for i in range(0,max_try):
        print('-------- computing try %i for dimension %i -----------' %(i,ndim))
        filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
            
        f3 = np.load(filename3)
        sol3 = f3['sol3']
        x3 = f3['x3']
        x3_store[:,i] = x3.copy()


        ### computes values of adaptive/global smoothing at sols
        
        ii = 0
        for entry in sol3:
            if abs(entry[0]) < opt3.eps_val:
                opt3.npop = opt3.npop_stoch
                opt3.gradient_available = False
                Fsol3_store[ii,i] = opt3.compute_val(entry)
                opt3.npop = opt3.npop_der
                opt3.gradient_available = True
            else:
                Fsol3_store[ii,i] = opt3.compute_val(entry)
            ii+=1
        ii = 0


        filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
        
        f1 = np.load(filename1)
        sol1 = f1['sol1']
        x1 = f1['x1']
        x1_store[:,i] = x1.copy()

        ii = 0
        for entry in sol1:
            Fsol1_store[ii,i] = opt1.compute_val(entry)
            ii+=1
    
    with open(pathing + '/Vals_hybrid.npy', 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store[:,:], x3_store = x3_store[:,:])
    with open(pathing + '/Vals_stoch.npy', 'wb') as f:
        np.savez(f, Fsol1_store = Fsol1_store[:,:], x1_store = x1_store[:,:])

    
    k+=1

    return


















