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
from Gradient_Descent_variable_circle import GradientDescent

xmin1 = [0,0]
xmin2 = [-3,3]

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
            vals.append( np.abs(entry[0]) + np.abs(entry[1]) +np.abs(entry[1]+entry[0]) + 2*evalue
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

fmin = Spring_fct
fmincon = lambda x: Spring_fct(x,['Derivative'])


date_time = 'final_circ_new'
pathing = './Ex_nGs_' + date_time 
metadata = pathing + '/metadata.npy'
if not os.path.exists(pathing): 
    os.makedirs(pathing)
    
### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']

do_work = True
do_work = False

### if yes, will compute function values of the simulation
### if no, will load them from an (assumed to be) existing file instead
if do_work:
    
    ### initialiize data storage vectors
    Fsol1_ex_store = np.zeros((iter_max,max_try))
    x1_store = np.zeros(((iter_max,max_try)))
    Fsol3_ex_store = np.zeros((iter_max*multi,max_try))
    x3_store = np.zeros(((iter_max*multi,max_try)))


    pop_per_dir = 10
    sigma = 1
    eps_val = [np.sqrt(9)]
    kappa = [np.sqrt(4)]
    aleph = 5

    opt1 = GradientDescent(
        xstart=[0, 0], npop= 100, sigma=sigma, gradient_available = False, 
        npop_der = 1,npop_stoch = 10, num_id_funcs = 0
    )
    

    ### start optimization 
    opt1.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)

    opt3 = GradientDescent(
            xstart=[0, 0], npop=1, sigma=sigma, gradient_available = True, 
            check_for_stability = True, use_one_directional_smoothing = True,
            npop_der = 1,npop_stoch = pop_per_dir, num_id_funcs = 2
        )
        

    ### start optimization to initialize for a single step
    opt3.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    

    for i in range(0,max_try):
        print('-------- computing try %i -----------' %i)
        filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
        filename2 = pathing + '/Run_' + str(i) + '_results2.npy'
        filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
            
        f3 = np.load(filename3)
        sol3 = f3['sol3']
        x3 = f3['x3']
        x3_store[:,i] = x3.copy()

        
        f1 = np.load(filename1)
        sol1 = f1['sol1']
        x1 = f1['x1']
        x1_store[:,i] = x1.copy()


        ### computes values of adaptive/global smoothing at sols
        
        ii = 0
        opt3.gradient_available = False
        for entry in sol3:
            Fsol3_ex_store[ii,i] = opt3.compute_val(entry)
            ii+=1
        ii = 0
        for entry in sol1:
            Fsol1_ex_store[ii,i] = opt1.compute_val(entry)
            ii+=1

    Fsol1_store = Fsol1_ex_store
    Fsol3_store = Fsol3_ex_store

    with open(pathing + '/Vals_stoch.npy', 'wb') as f:
        np.savez(f, Fsol1_store = Fsol1_store, x1_store = x1_store)

    with open(pathing + '/Vals_hybrid.npy', 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store, x3_store = x3_store)
        
else:
    filename_stoch = pathing + '/Vals_stoch.npy'
    filename_hybrid = pathing + '/Vals_hybrid.npy'
    f_stoch = np.load(filename_stoch)
    f_hyb = np.load(filename_hybrid)
    x1_store = f_stoch['x1_store']
    Fsol1_store = f_stoch['Fsol1_store']
    x3_store = f_hyb['x3_store']
    Fsol3_store = f_hyb['Fsol3_store']
    

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

max_store = max(x3_store_list[-1],x1_store_list[-1],2*10**3 )
x3_store_list=np.append(x3_store_list,max_store)
x1_store_list=np.append(x1_store_list,max_store)
Fsol3_store_list=np.append(Fsol3_store_list,Fsol3_store_list[-1])
Fsol1_store_list=np.append(Fsol1_store_list,Fsol1_store_list[-1])
variance3_list=np.append(variance3_list,variance3_list[-1])
variance1_list=np.append(variance1_list,variance1_list[-1])


ax4[-1].plot(x3_store_list,Fsol3_store_list, label = 'Mean Objective Value', color = 'tab:blue')
ax4[-1].fill_between(x3_store_list, Fsol3_store_list - np.sqrt(variance3_list), Fsol3_store_list + np.sqrt(variance3_list), alpha = 0.3, label = 'Standard Deviation', color = 'tab:blue')
ax4[-1].fill_between(x3_store_list[:int(x3_store_0)], (Fsol3_store_list - np.sqrt(variance3_list))[:int(x3_store_0)], (Fsol3_store_list + np.sqrt(variance3_list))[:int(x3_store_0)], alpha = 0.3, color = 'tab:blue', hatch = '///')

ax4[0].plot(x1_store_list,Fsol1_store_list, label = 'Mean Objective Value', color = 'tab:blue')
ax4[0].fill_between(x1_store_list, Fsol1_store_list - np.sqrt(variance1_list), Fsol1_store_list + np.sqrt(variance1_list), alpha = 0.3, label = 'Standard Deviation', color = 'tab:blue')
ax4[0].fill_between(x1_store_list[:int(x1_store_0)], (Fsol1_store_list - np.sqrt(variance1_list))[:int(x1_store_0)], (Fsol1_store_list + np.sqrt(variance1_list))[:int(x1_store_0)], alpha = 0.3, color = 'tab:blue', hatch = '///')

upper_lim = np.max([ np.max(Fsol1_store_list), np.max(Fsol3_store_list)]) + max(np.sqrt(variance3_list)[0],np.sqrt(variance1_list)[0])
lower_lim = np.min([ np.min(Fsol1_store_list), np.min(Fsol3_store_list)]) 

iterations_bound = int( x3_store_0)*iter_max
iterations_bound = x3_store_list[-1]
iterations_bound = 2*10**4

ax4[-1].set_xlabel("Function Evaluations", fontsize = 12)
ax4[-1].set_xscale("log")
ax4[-1].set_ylim([lower_lim-7, upper_lim+2])
ax4[-1].set_xlim([1,iterations_bound])
ax4[-1].set_ylabel("ASM", fontsize = 12)

ax4[0].set_xscale("log")
ax4[0].set_ylim([lower_lim-7, upper_lim+2])
ax4[0].set_xlim([1,iterations_bound])
ax4[0].set_ylabel("GSM", fontsize = 12)

m = 10### set to number of directional samples
fig4.suptitle('Mean and Standard Deviation in Objective Value for %i random runs' %max_try)
ax4[-1].legend()
plt.subplots_adjust(hspace=0.2)

print('Final Function Values: \n Global: %f \n Adaptive: %f' %(Fsol1_store_list[-1], Fsol3_store_list[-1]))
print('Final Standard Deviation Values: \n Global: %f \n Adaptive: %f' %(np.sqrt(variance1_list)[-1], np.sqrt(variance3_list)[-1] ) )

conv1_b = np.argmax(Fsol1_store == Fsol1_store[-1,:], axis = 0)
conv3_b = np.argmax(Fsol3_store == Fsol3_store[-1,:], axis = 0)
conv1_c = int(np.average(conv1_b))
conv3_c = int(np.average(conv3_b))
print('Average Number of Function Evaluations until Convergence: \n Global: %f \n Adaptive: %f' %(np.average(x1_store[conv1_b,range(len(conv1_b))]), np.average(x3_store[conv3_b,range(len(conv3_b))]) ))
print('Weighted Average Number of Function Evaluations until Convergence: \n Global: %f \n Adaptive: %f' %(np.sum(x1_store[conv1_b,range(len(conv1_b))])/np.sum(conv1_b), np.sum(x3_store[conv3_b,range(len(conv3_b))])/np.sum(conv3_b) ))
print('Average Number of Steps until Convergence: \n Global: %f \n Adaptive: %f' %(np.average(conv1_b), np.average(conv3_b) ))


plt.show()


















