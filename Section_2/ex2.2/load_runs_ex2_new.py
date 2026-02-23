############################################
#### Script to load the generated data from your run of the gd_ex3.py script
#### and postprocess them, so theyre visually more appealing.
#### This script is responsible for visualizing the compuational cost of different methods
#### and plotting them side by side to compare them 
############################################

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.tri as tri
import datetime
import scipy.interpolate as sp
from gd_ex2_new import GradientDescent

xmin1 = [0,0]
xmin2 = [-3,3]

def abs_sphere(x, args=()):
    ### discontinuous absolute value function, with instant drop at x= 0
    x = np.reshape(x, (-1, x[0].shape[-1]))
    f_x = np.where(
        x[:, 0] > 0.0,
        1 + 0.1*np.absolute(x - xmin1)[:,0] + 0.01 * (x - xmin1)[:,1]**2,
        -1 + 0.1*np.absolute(x - xmin1)[:,0] + 0.01 * (x - xmin1)[:,1]**2,
    )
    return f_x

def abs_gradient(x, args=()):
    ### gradient function for discontinuous absolute value function
    x = np.reshape(x, (-1, x[0].shape[-1]))
    df_x = np.zeros(x.shape)        
    df_x[np.where(x[:, 0] > 0.0),0] = 0.1 
    df_x[np.where(x[:, 0] <= 0.0),0] = -0.1 
    df_x[:,1] = 0.02 * (x-xmin1)[:,1]
    return df_x

fmin = abs_sphere
grad_min = abs_gradient
## path to dataset
date_time = 'final_new'
pathing = './Ex2_' + date_time 
metadata = pathing + '/metadata.npy'
if not os.path.exists(pathing): 
    os.makedirs(pathing)

### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']
print('metadata:', max_try, iter_max, multi)


do_work = True
do_work = False

### if yes, will compute function values of the simulation
### if no, will load them from an (assumed to be) existing file instead
if do_work:
    
    ### initialiize data storage vectors
    Fsol1_ex_store = np.zeros((iter_max,max_try))
    x1_store = np.zeros(((iter_max,max_try)))
    sol1_store = np.zeros( (iter_max,2, max_try) )

    Fsol3_ex_store = np.zeros((iter_max*multi,max_try))
    x3_store = np.zeros(((iter_max*multi,max_try)))
    sol3_store = np.zeros( (iter_max*multi,2, max_try) )

    npop_val = 100
    npop_val_1d = 10

    opt1 = GradientDescent( xstart = [-1,1], npop = npop_val, sigma = 1, gradient_available = False)
    opt3 = GradientDescent( xstart = [-1,1], npop = 1, sigma = 1, gradient_available = True,
                       check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,
                       npop_stoch = npop_val_1d)
    opt1.optimize(fmin, iterations= 1, gradient_function = grad_min)
    opt3.optimize(fmin, iterations= 1, gradient_function = grad_min)

    for i in range(0,max_try):
        print('-------- computing try %i -----------' %i)
        filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
        filename2 = pathing + '/Run_' + str(i) + '_results2.npy'
        filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
            
        f3 = np.load(filename3)
        sol3 = f3['sol3']
        x3 = f3['x3']
        x3_store[:,i] = x3.copy()
        sol3_store[:,:,i] = sol3.copy()
        
        f1 = np.load(filename1)
        sol1 = f1['sol1']
        x1 = f1['x1']
        x1_store[:,i] = x1.copy()
        sol1_store[:,:,i] = sol1.copy()


        ### computes values of adaptive/global smoothing at sols
        
        ii = 0
        for entry in sol3:
            if abs(entry[0])**2 < opt3.eps_val:
                opt3.npop = opt3.npop_stoch
                opt3.gradient_available = False
                Fsol3_ex_store[ii,i] = opt3.compute_val(entry)
                opt3.npop = opt3.npop_der
                opt3.gradient_available = True
            else:
                Fsol3_ex_store[ii,i] = opt3.compute_val(entry)
            ii+=1
        ii = 0
        for entry in sol1:
            Fsol1_ex_store[ii,i] = opt1.compute_val(entry)
            ii+=1

    Fsol1_store = Fsol1_ex_store
    Fsol3_store = Fsol3_ex_store

    with open(pathing + '/Vals_stoch.npy', 'wb') as f:
        np.savez(f, Fsol1_store = Fsol1_store, x1_store = x1_store, sol1_store = sol1_store)

    with open(pathing + '/Vals_hybrid.npy', 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store, x3_store = x3_store, sol3_store = sol3_store)
        
else:
    filename_stoch = pathing + '/Vals_stoch.npy'
    filename_hybrid = pathing + '/Vals_hybrid.npy'
    f_stoch = np.load(filename_stoch)
    f_hyb = np.load(filename_hybrid)
    x1_store = f_stoch['x1_store']
    Fsol1_store = f_stoch['Fsol1_store']
    sol1_store = f_stoch['sol1_store']
    x3_store = f_hyb['x3_store']
    Fsol3_store = f_hyb['Fsol3_store']
    sol3_store = f_hyb['sol3_store']
    

### plots the number of function evaluation vs the function value 
fig4,ax4 = plt.subplots(nrows = 2, sharex = True)

print('############################  plotting now   ########################################')

x1_store_0 = np.average(x1_store, axis = 1)[0]
x1_store_list = np.append( range(1,int(x1_store_0)), np.average(x1_store, axis = 1))
Fsol1_store_list = np.append( np.average(Fsol1_store, axis = 1)[0] * np.ones((int(x1_store_0)-1,1)), np.average(Fsol1_store, axis = 1))

x3_store_0 = np.average(x3_store, axis = 1)[0]
if int(x3_store_0) == 1:
    x3_store_list = np.append( [1], np.average(x3_store, axis = 1))
    Fsol3_store_list = np.append( np.average(Fsol3_store, axis = 1)[0] * np.ones((int(x3_store_0),1)), np.average(Fsol3_store, axis = 1))
    variance3 = abs(np.average(Fsol3_store**2, axis = 1) - np.average(Fsol3_store, axis = 1)**2)
    variance3_list = np.append( variance3[0] * np.ones((int(x3_store_0),1)), variance3)
else:
    x3_store_list = np.append( range(1,int(x3_store_0)), np.average(x3_store, axis = 1))
    Fsol3_store_list = np.append( np.average(Fsol3_store, axis = 1)[0] * np.ones((int(x3_store_0)-1,1)), np.average(Fsol3_store, axis = 1))
    variance3 = abs(np.average(Fsol3_store**2, axis = 1) - np.average(Fsol3_store, axis = 1)**2)
    variance3_list = np.append( variance3[0] * np.ones((int(x3_store_0)-1,1)), variance3)

variance1 = abs(np.average(Fsol1_store**2, axis = 1) - np.average(Fsol1_store, axis = 1)**2)
variance1_list = np.append( variance1[0] * np.ones((int(x1_store_0)-1,1)), variance1)

sol1_store_ext = np.zeros(sol3_store.shape)
sol1_store_ext[:iter_max,:,:] = sol1_store
sol1_store_ext[iter_max:iter_max*multi,:,:] = sol1_store[-1,:,:] * np.ones( (iter_max*(multi-1),2,1))
soldiff = np.linalg.norm(sol1_store_ext - sol3_store, axis = 1)
soldiff_m = np.average(soldiff, axis = 1)
variance_sol = abs(np.average(soldiff**2, axis = 1) - np.average(soldiff, axis = 1)**2)


ax4[-1].plot(x3_store_list,Fsol3_store_list, label = 'Mean Objective Value', color = 'tab:blue')
ax4[-1].fill_between(x3_store_list, Fsol3_store_list - np.sqrt(variance3_list), Fsol3_store_list + np.sqrt(variance3_list), alpha = 0.3, label = 'Standard Deviation', color = 'tab:blue')
ax4[-1].fill_between(x3_store_list[:int(x3_store_0)+1], (Fsol3_store_list - np.sqrt(variance3_list))[:int(x3_store_0)+1], (Fsol3_store_list + np.sqrt(variance3_list))[:int(x3_store_0)+1], alpha = 0.3, color = 'tab:blue', hatch = '///')

ax4[0].plot(x1_store_list,Fsol1_store_list, label = 'Mean Objective Value', color = 'tab:blue')
ax4[0].fill_between(x1_store_list, Fsol1_store_list - np.sqrt(variance1_list), Fsol1_store_list + np.sqrt(variance1_list), alpha = 0.3, label = 'Standard Deviation', color = 'tab:blue')
ax4[0].fill_between(x1_store_list[:int(x1_store_0)], (Fsol1_store_list - np.sqrt(variance1_list))[:int(x1_store_0)], (Fsol1_store_list + np.sqrt(variance1_list))[:int(x1_store_0)], alpha = 0.3, color = 'tab:blue', hatch = '///')

upper_lim = np.max([ np.max(Fsol1_store_list), np.max(Fsol3_store_list)]) + max(np.sqrt(variance3_list)[0],np.sqrt(variance1_list)[0])
lower_lim = np.min([ np.min(Fsol1_store_list), np.min(Fsol3_store_list)]) 

iterations_bound = int( x3_store_0)*iter_max
iterations_bound = x3_store_list[-1]
iterations_bound = 7*10**5

ax4[-1].set_xlabel("Function Evaluations", fontsize = 12)
ax4[-1].set_xscale("log")
ax4[-1].set_ylim([lower_lim-0.5, upper_lim+0.2])
ax4[-1].set_xlim([1,iterations_bound])
ax4[-1].set_ylabel("ASM", fontsize = 12)

ax4[0].set_xscale("log")
ax4[0].set_ylim([lower_lim-0.5, upper_lim+0.2])
ax4[0].set_xlim([1,iterations_bound])
ax4[0].set_ylabel("GSM", fontsize = 12)

m = 10### set to number of directional samples
fig4.suptitle('Mean and Standard Deviation in Objetive Value for %i random runs' %max_try)
ax4[-1].legend()
plt.subplots_adjust(hspace=0.2)

print('Final Function Values: \n Global: %f \n Adaptive: %f' %(Fsol1_store_list[-1], Fsol3_store_list[-1]))
print('Final Standard Deviation Values: \n Global: %f \n Adaptive: %f' %(np.sqrt(variance1_list)[-1], np.sqrt(variance3_list)[-1] ) )
conv1_b = np.argmax(Fsol1_store == Fsol1_store[-1,:], axis = 0)
conv3_b = np.argmax(Fsol3_store == Fsol3_store[-1,:], axis = 0)
conv1_c = int(np.average(conv1_b))
conv3_c = int(np.average(conv3_b))
print('Average Number of Function Evaluations until Convergence: \n Global: %f \n Adaptive: %f' %(np.average(x1_store[conv1_b,range(len(conv1_b))]), np.average(x3_store[conv3_b,range(len(conv3_b))]) ))
print('Weighted Average Number of Function Evaluations per step until Convergence: \n Global: %f \n Adaptive: %f' %(np.sum(x1_store[conv1_b,range(len(conv1_b))])/np.sum(conv1_b), np.sum(x3_store[conv3_b,range(len(conv3_b))])/np.sum(conv3_b) ))
print('Average Number of Steps until Convergence: \n Global: %f \n Adaptive: %f' %(np.average(conv1_b), np.average(conv3_b) ))
plt.show()


















