import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


pathing = './ExAS_' +'final_new'
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

if do_work:
    Fsol3_store_full = np.zeros((iter_max*multi,max_try))
    x3_store_full = np.zeros((iter_max*multi,max_try))

    Fsol1_store_full = np.zeros((iter_max,max_try))
    x1_store_full = np.zeros((iter_max,max_try))
    
    for i in range(max_try):
        k = i
        filename_stoch = pathing +'/Run_'+ str(k) + 'Vals_stoch.npy'
        filename_hybrid = pathing +'/Run_'+ str(k) + 'Vals_hybrid.npy'
        f_stoch = np.load(filename_stoch)
        f_hyb = np.load(filename_hybrid)
        x1_store_full[:,i] = f_stoch['x1_store'][:,k]
        Fsol1_store_full[:,i] = f_stoch['Fsol1_store'][:,k]
        x3_store_full[:,i] = f_hyb['x3_store'][:,k]
        Fsol3_store_full[:,i] = f_hyb['Fsol3_store'][:,k]

    filename_stoch = pathing + '/Vals_stoch.npy'
    filename_hybrid = pathing + '/Vals_hybrid.npy'
    with open(filename_hybrid, 'wb') as f:
        np.savez(f, Fsol3_store = Fsol3_store_full, x3_store = x3_store_full)
    with open(filename_stoch, 'wb') as f:
        np.savez(f, Fsol1_store = Fsol1_store_full, x1_store = x1_store_full)
    
        

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

max_store = max(x3_store_list[-1],x1_store_list[-1] )
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
iterations_bound = 3*10**3

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
##ax4[-1].set_title('%i samples per evaluation near discontinuity'%(m))
##ax4[0].set_title('%i samples per evaluation'%(m**2))
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
