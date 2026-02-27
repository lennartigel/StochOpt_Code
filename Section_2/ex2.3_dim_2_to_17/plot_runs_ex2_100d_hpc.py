############################################
#### Script to load the generated data from your run 
#### and postprocess them, so theyre visually more appealing.
#### This script is responsible for visualizing the compuational cost of different methods
#### and plotting them side by side to compare them 
############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import datetime
import scipy.interpolate as sp
from gd_ex2_100d_hpc import GradientDescent

ndim = 100


dim_list = range(2,18)
num_dims = len(dim_list)

pathfront = './comp_small_new/final_2_to_17_new_' ### path to dataset
pathing = pathfront + str(dim_list[0])
metadata = pathing + '/metadata.npy'

### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']
                
Fsol3_store = np.zeros((iter_max*multi,max_try,num_dims ))
x3_store = np.zeros(((iter_max*multi,max_try ,num_dims)))

Fsol1_store = np.zeros((iter_max,max_try,num_dims ))
x1_store = np.zeros(((iter_max,max_try ,num_dims)))
    
k = 0
for ndim in dim_list:
    pathing = pathfront + str(ndim)
    filename_hybrid = pathing + '/Vals_hybrid.npy'
    f_hyb = np.load(filename_hybrid)
    x3_store[:,:,k] = f_hyb['x3_store']
    Fsol3_store[:,:,k] = f_hyb['Fsol3_store']
    
    filename_stoch = pathing + '/Vals_stoch.npy'
    f_stoch = np.load(filename_stoch)
    x1_store[:,:,k] = f_stoch['x1_store']
    Fsol1_store[:,:,k] = f_stoch['Fsol1_store']
    k+=1
    

### plots the number of function evaluation vs the function value
fig3,ax3 = plt.subplots(nrows = 2, sharex = True)
fig4,ax4 = plt.subplots(nrows = 2, sharex = True)

print('############################  plotting now  ########################################')

x3_store_list = []
x3_store_0_list = []
Fsol3_store_list = []
variance3_list = []
x1_store_list = []
x1_store_0_list = []
Fsol1_store_list = []
variance1_list = []
x1_conv_list = []
x3_conv_list = []
x1_conv_list_norm = []
x3_conv_list_norm = []
for k in range(num_dims):
    x1_store_0 = np.average(x1_store[:,:,k], axis = 1)[0]
    x1_store_0_list.append(x1_store_0)
    x1_store_list.append( np.append( range(1,int(x1_store_0)), np.average(x1_store[:,:,k], axis = 1)) )
    Fsol1_store_list.append(np.append( np.average(Fsol1_store[:,:,k], axis = 1)[0] * np.ones((int(x1_store_0)-1,1)), np.average(Fsol1_store[:,:,k], axis = 1)) )
    variance1 = np.sqrt(abs(np.average(Fsol1_store[:,:,k]**2, axis = 1) - np.average(Fsol1_store[:,:,k], axis = 1)**2))
    variance1_list.append(np.append( variance1[0] * np.ones((int(x1_store_0)-1,1)), variance1))

    x3_store_0 = np.average(x3_store[:,:,k], axis = 1)[0]
    x3_store_0_list.append(x3_store_0)
    print('dimension:',dim_list[k])
    if int(x3_store_0) == 1:
        x3_store_list.append(np.append( [1], np.average(x3_store[:,:,k], axis = 1)))
        Fsol3_store_list.append(np.append( np.average(Fsol3_store[:,:,k], axis = 1)[0] * np.ones((int(x3_store_0),1)), np.average(Fsol3_store[:,:,k], axis = 1)) )
        variance3 = np.sqrt(abs(np.average(Fsol3_store[:,:,k]**2, axis = 1) - np.average(Fsol3_store[:,:,k], axis = 1)**2))
        variance3_list.append( np.append( variance3[0] * np.ones((int(x3_store_0),1)), variance3) )
    else:
        x3_store_list.append( np.append( range(1,int(x3_store_0)), np.average(x3_store[:,:,k], axis = 1)) )
        Fsol3_store_list.append( np.append( np.average(Fsol3_store[:,:,k], axis = 1)[0] * np.ones((int(x3_store_0)-1,1)), np.average(Fsol3_store[:,:,k], axis = 1)) )
        variance3 = np.sqrt(abs(np.average(Fsol3_store[:,:,k]**2, axis = 1) - np.average(Fsol3_store[:,:,k], axis = 1)**2))
        variance3_list.append( np.append( variance3[0] * np.ones((int(x3_store_0)-1,1)), variance3) )

for k in range(num_dims):

    
    m = 15### set to number of directional samples
    m2 = 3

    ax3[0].plot(x1_store_list[k],variance1_list[k], label = 'dim = %i'%dim_list[k] )
    ax3[-1].plot(x3_store_list[k],variance3_list[k], label = 'dim = %i'%dim_list[k] )
    ax3[0].set_ylabel("GSM", fontsize = 12)
    ax3[-1].set_ylabel("ASM", fontsize = 12)
    ax3[-1].set_xlabel("Function Evaluations", fontsize = 12)
    ax3[-1].set_xscale("log")
    ax3[0].set_yscale("log")
    ax3[-1].set_yscale("log")
    ax3[-1].set_xlim(left = 1, right = x1_store_list[-1][-1] )
    fig3.suptitle(r'Standard Deviation for %i random runs with increasing dimension' %max_try)

    iterations_bound = 1000

    ax4[-1].plot(x3_store_list[k],Fsol3_store_list[k], label = 'dim = %i'%dim_list[k] )
    ax4[0].plot(x1_store_list[k],Fsol1_store_list[k], label = 'dim = %i'%dim_list[k] )
    ax4[0].set_ylabel("GSM", fontsize = 12)
    ax4[-1].set_xlabel("Function Evaluations", fontsize = 12)
    ax4[-1].set_xscale("log")
    ax4[0].set_yscale("log")
    ax4[-1].set_yscale("log")
    ax4[-1].set_xlim(left = 1, right = x1_store_list[-1][-1] )
    ax4[-1].set_ylabel("ASM", fontsize = 12)
    fig4.suptitle('Mean Objective Value for %i random runs with increasing dimension' %max_try)
    ax4[0].set_yscale("symlog", linthresh=1e-3)
    ax4[1].set_yscale("symlog", linthresh=1e-3)




    
plt.subplots_adjust(hspace=0.2)


print('Final Function Values Adaptive:')
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], Fsol3_store_list[k][-1] ) )
print('Final Standard Deviation Values Adaptive:' )
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], (variance3_list[k])[-1] ) )
print('Average Cost of an Optimization Step Adaptive:')
val_tot = np.zeros(num_dims)
val_av = np.zeros(num_dims)
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    val_av[k] = np.sum(x3_store[conv3_b,range(len(conv3_b)),k])/np.sum(conv3_b)
    print(' dim %i: %f' %( dim_list[k], val_av[k] ) )
print('Number of Function Evaluations until Convergence Adaptive:' )
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    val_tot[k] = np.average(x3_store[conv3_b,range(len(conv3_b)),k])
    
    conv3 = np.where(Fsol3_store_list[k] == Fsol3_store_list[k][-1])[0][0]
    x3_conv_list.append(val_tot[k])
    x3_conv_list_norm.append( val_av[k])
    print(' dim %i: %f' %( dim_list[k], val_tot[k] ) )

print('Average Number of Optimization Steps Adaptive:')
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    print(' dim %i: %f' %( dim_list[k], np.average(conv3_b) ) )

print(r'Final $\mathbf{x}_1$ Values Global:')
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], sol1_store_list[k][-1] ) )
print('Final $\mathbf{x}_1$ Standard Deviation Values Global:' )
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], (variance_s1_list[k])[-1] ) )
print('Final Function Values Global:')
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], Fsol1_store_list[k][-1] ) )
print('Final Standard Deviation Values Global:' )
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], (variance1_list[k])[-1] ) )

val_tot_1 = np.zeros(num_dims)
val_av_1 = np.zeros(num_dims)
print('Average Cost of an Optimization Step Global:')
for k in range(num_dims):
    conv1_b = np.argmax(Fsol1_store[:,:,k] == Fsol1_store[-1,:,k], axis = 0)
    conv1_c = int(np.average(conv1_b))
    x1_store[:,:,k]
    val_av_1[k] = np.sum(x1_store[conv1_b,range(len(conv1_b)),k])/np.sum(conv1_b)
    print(' dim %i: %f' %( dim_list[k], val_av_1[k] ) )
print('Number of Function Evaluations until Convergence Global:' )
for k in range(num_dims):
    conv1_b = np.argmax(Fsol1_store[:,:,k] == Fsol1_store[-1,:,k], axis = 0)
    conv1_c = int(np.average(conv1_b))
    val_tot_1[k] = np.average(x1_store[conv1_b,range(len(conv1_b)),k])
    print(' dim %i: %f' %( dim_list[k], val_tot_1[k] ) )
    x1_conv_list.append(val_tot_1[k])
    x1_conv_list_norm.append( val_av_1[k])

print('Average Number of Optimization Steps Global:')
for k in range(num_dims):
    conv1_b = np.argmax(Fsol1_store[:,:,k] == Fsol1_store[-1,:,k], axis = 0)
    conv1_c = int(np.average(conv1_b))
    print(' dim %i: %f' %( dim_list[k], np.average(conv1_b) ) )


fig9,ax9 = plt.subplots(nrows = 2, sharex = True)
fig9.suptitle('Mean total cost until convergence and mean cost per step')
##ax9[0].plot(dim_list, x1_conv_list_norm, label = 'cost per step')
ax9[0].plot(dim_list, x1_conv_list_norm, label = r'$c_\mathrm{mean}^\mathrm{step}$')
##ax9[0].plot(dim_list, x1_conv_list, label = 'total cost')
ax9[0].plot(dim_list, x1_conv_list, label = r'$c_\mathrm{mean}$')
ax9[1].plot(dim_list, x3_conv_list_norm,  label = r'$c_\mathrm{mean}^\mathrm{step}$')
ax9[1].plot(dim_list, x3_conv_list,  label = r'$c_\mathrm{mean}$')


ax9[1].set_xlim(left = dim_list[0], right = dim_list[-1] )
ax9[1].minorticks_on()
ax9[0].minorticks_on()
ax9[0].tick_params(axis='y', which='both', left=True)
ax9[0].set_yscale("log")
ax9[1].set_yscale("log")
ax9[1].minorticks_on()
ax9[0].minorticks_on()
ax9[0].set_ylabel("GSM", fontsize = 12)
ax9[1].set_ylabel("ASM", fontsize = 12)
ax9[1].set_xlabel(r"Dimension $N$", fontsize = 12)
ax9[0].legend(loc = 'upper left',fontsize = 12 )
ax9[1].legend(loc = 'center left',fontsize = 12)



plt.show()


















