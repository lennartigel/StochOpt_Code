############################################
#### Script to generate data for landscapes
############################################


import numpy as np
import os
import scipy.interpolate as sp
from Springclass_hpc_AS import FractureProblem
from gd_ex_spring_hpc_new import GradientDescent



def run_fun(jj):

    method = 'staggered_multi_step'

    directory = './LS/'
    
##    timecode = 'final_spring'
    timecode = 'test' ### so nothing is overwritten by accident
    code = timecode + str(jj)

    filename_ls_r = './LS/'  + code + '_Regular_Landscape.npy'
    filename_ls_g = './LS/'  + code + '_Glob_Landscape.npy'
    filename_ls_ad = './LS/'  + code + '_Adap_Landscape.npy'




    n_time_steps = 101
    endtime = 1
    Springprob = FractureProblem(n_time_steps, endtime)

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


    ### define objective functions
    fmin = Spring_fct
    fmincon = lambda x: Spring_fct(x,['Derivative'])

    sigma = 0.15*1.0 ### size of variance for normal distribution area for sampling
    eps_val = np.sqrt(0.7)
    kappa = np.sqrt(0.5)
    aleph = 0.2


    npop_stoch_num = 10

    opt1 = GradientDescent(
        xstart=[0.7, 0.9], npop= npop_stoch_num**2, sigma=sigma, gradient_available = False, 
        npop_der = 1,npop_stoch = npop_stoch_num
    )
    opt3 = GradientDescent(
                xstart=[0.7, 0.9], npop=1, sigma=sigma, gradient_available = False, 
                check_for_stability = True, use_one_directional_smoothing = True,
                npop_der = 1,npop_stoch = npop_stoch_num
            )
    opt1.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    opt3.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)



    vals = []
    vals_adap = []
    vals_stoch = []
    eigs = []


    num_samples = 128 ### set resolution 

    if do_work:

        add = 0
        xx = add + np.linspace(0.2,2,num_samples)
        yy = xx + 0.001 

        x_1, x_2 = np.meshgrid(xx, yy, indexing="ij")
        
        for i in range( jj*int(num_samples**2/32), (jj+1)*int(num_samples**2/32)):

            print('starting iteration ',i)
            (ff,ff_grad,ff_E) = fmin([[x_1.ravel()[i],x_2.ravel()[i]]], [9,9]) ### value, grad value and G value at point

            ### check whether smoothing for ASM value
            if ff_E**2 < eps_val**2: 
                opt3.gradient_available = False
                opt3.npop = npop_stoch_num
                ff_ad = opt3.compute_val(np.asarray([x_1.ravel()[i],x_2.ravel()[i]]))
                dE = opt3.compute_smoothing_direction(np.asarray([x_1.ravel()[i],x_2.ravel()[i]]))
                direction  = dE[-1]/np.linalg.norm(dE[-1])

            else:
                opt3.gradient_available = True
                opt3.npop = 1
                ff_ad = ff
                direction = np.zeros((2,))

            ### GSM value    
            ff_st = opt1.compute_val([x_1.ravel()[i],x_2.ravel()[i]])

            ### store values in list
            vals.append(ff)
            vals_adap.append(ff_ad)
            vals_stoch.append(ff_st)
            eigs.append(ff_E)
            grads.append(direction)
            
            print('done iteration ',i)


        vals_adap = np.asarray(vals_adap)
        vals_stoch = np.asarray(vals_stoch)
        vals = np.asarray(vals)
        eigs = np.asarray(eigs)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filename_ls_r, 'wb') as f:
            np.savez(f, eigs = eigs, vals = vals, x_1 = x_1, x_2 = x_2) ### regular
        with open(filename_ls_ad, 'wb') as f:
            np.savez(f, vals_adap = vals_adap, grads = grads, x_1 = x_1, x_2 = x_2, eps_val = eps_val, kappa = kappa) ### adaptive
        with open(filename_ls_g, 'wb') as f:
            np.savez(f, vals_stoch = vals_stoch, x_1 = x_1, x_2 = x_2) ### global


