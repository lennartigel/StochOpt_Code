import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


pathing = './Ex_QSAF_final'

metadata = pathing + '/metadata.npy'
if not os.path.exists(pathing): 
    os.makedirs(pathing)

### load metadata
meta = np.load(metadata)
max_try = meta['max_try']### this is not the number of runs performed here. Its a fixed number set in each run for the metadata, to be assigned an id, but we only did 100 runs
iter_max = meta['iter_max']
multi = meta['multi']
print('metadata:', max_try, iter_max, multi)


max_tried = 100 ### actual number of runs performed

do_work = True
do_work = False ### so nothing is accidentally overwritten

if do_work:
    Fsol3_store_full = np.zeros((iter_max*multi,max_try))
    x3_store_full = np.zeros((iter_max*multi,max_try))
    
    for i in range(max_tried):
        k = i
        filename_hybrid = pathing +'/Run_'+ str(k) + 'Vals_hybrid.npy'
        f_hyb = np.load(filename_hybrid)
        x3_store_full[:,i] = f_hyb['x3_store'][:,k]
        Fsol3_store_full[:,i] = f_hyb['Fsol3_store'][:,k]
        filename_hybrid = pathing + '/Vals_hybrid.npy'
        with open(filename_hybrid, 'wb') as f:
            np.savez(f, Fsol3_store = Fsol3_store_full, x3_store = x3_store_full)
        

filename_hybrid = pathing + '/Vals_hybrid.npy'
f_hyb = np.load(filename_hybrid)

x3_store = f_hyb['x3_store']
Fsol3_store = f_hyb['Fsol3_store']
    


### plots the number of function evaluation vs the function value 
fig4,ax4 = plt.subplots(nrows = 1, sharex = True)

print('############################  plotting now   ########################################')

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


max_store = max(x3_store_list[-1],0 )
x3_store_list=np.append(x3_store_list,max_store)
Fsol3_store_list=np.append(Fsol3_store_list,Fsol3_store_list[-1])
variance3_list=np.append(variance3_list,variance3_list[-1])

ax4.plot(x3_store_list,Fsol3_store_list, label = 'Mean Objective Value', color = 'tab:blue')
ax4.fill_between(x3_store_list, Fsol3_store_list - np.sqrt(variance3_list), Fsol3_store_list + np.sqrt(variance3_list), alpha = 0.3, label = 'Standard Deviation', color = 'tab:blue')
ax4.fill_between(x3_store_list[:int(x3_store_0)], (Fsol3_store_list - np.sqrt(variance3_list))[:int(x3_store_0)], (Fsol3_store_list + np.sqrt(variance3_list))[:int(x3_store_0)], alpha = 0.3, color = 'tab:blue', hatch = '///')

upper_lim = np.max([ 0, np.max(Fsol3_store_list + np.sqrt(variance3_list)) ])
lower_lim = np.min([ 0, np.min(Fsol3_store_list - np.sqrt(variance3_list)) ])

iterations_bound = x3_store_list[-1] + 0.1* x3_store_list[-1] 

ax4.set_xlabel("Function Evaluations", fontsize = 12)
ax4.set_xscale("log")
ax4.set_ylim([lower_lim-0.2, upper_lim+0.2])
ax4.set_xlim([1,iterations_bound])
ax4.set_ylabel("ASM", fontsize = 12)


m = 10### set to number of directional samples
fig4.suptitle('Mean and Standard Deviation in Objetive Value for %i random runs' %max_tried)
ax4.legend(fontsize = 12)
plt.subplots_adjust(hspace=0.2)

print('Final Function Values: \n Adaptive: %f' %( Fsol3_store_list[-1]))
print('Final Standard Deviation Values:  \n Adaptive: %f' %( np.sqrt(variance3_list)[-1] ) )
conv3_b = np.argmax(Fsol3_store == Fsol3_store[-1,:], axis = 0)
conv3_c = int(np.average(conv3_b))
print('Average Number of Function Evaluations until Convergence: \n Adaptive: %f' %( np.average(x3_store[conv3_b,range(len(conv3_b))]) ))
print('Weighted Average Number of Function Evaluations per step until Convergence:\n Adaptive: %f' %( np.sum(x3_store[conv3_b,range(len(conv3_b))])/np.sum(conv3_b) ))
print('Average Number of Steps until Convergence: \n Adaptive: %f' %( np.average(conv3_b) ))

plt.show()
