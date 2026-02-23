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
from scipy.optimize import curve_fit
from gd_ex2_100d import GradientDescent

dim_list = [2,10,50, 10**2,5*10**2, 10**3,5*10**3, 10**4, 5*10**4, 10**5]
num_dims = len(dim_list)

##path_front = './comp_big_new/Ex2_100d_'### dataset destination
path_front = './comp_big_new/test_'
pathing = path_front + str(dim_list[0])
metadata = pathing + '/metadata.npy'

### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']


### if yes, will compute function values of the simulation
### if no, will load them from an (assumed to be) existing file instead
do_work = True
do_work = False

plots_vals = True
##plots_vals = False

if do_work:
    k = 0

    for ndim in dim_list:

        xmin1 = np.zeros((ndim,))

        def abs_sphere(x, args=()):
            ### discontinuous absolute value function, with instant drop at x= 0
            x = np.reshape(x, (-1, x[0].shape[-1]))
            f_x = np.where(
                x[:, 0] > 0.0,
                1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
                -1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
            )
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

        Fsol3_store = np.zeros((iter_max*multi,max_try,num_dims ))
        x3_store = np.zeros(((iter_max*multi,max_try ,num_dims)))
        
        nsamples = 15

        opt3 = GradientDescent( xstart = np.ones((ndim,)), npop = 1, sigma = 1, gradient_available = True,
                           check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,
                           npop_stoch = nsamples)
        opt3.optimize(fmin, iterations= 1, gradient_function = grad_min)
        pathing = path_front + str(ndim)

        for i in range(0,max_try):
            print('-------- computing try %i for dimension %i -----------' %(i,ndim))
            filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
                
            f3 = np.load(filename3)
            sol3 = f3['sol3']
            x3 = f3['x3']
            x3_store[:,i,k] = x3.copy()


            ### computes values of adaptive/global smoothing at sols
            
            ii = 0
            for entry in sol3:
                if abs(entry[0]) < opt3.eps_val:
                    opt3.npop = opt3.npop_stoch
                    opt3.gradient_available = False
                    Fsol3_store[ii,i,k] = opt3.compute_val(entry)
                    opt3.npop = opt3.npop_der
                    opt3.gradient_available = True
                else:
                    Fsol3_store[ii,i,k] = opt3.compute_val(entry)
                ii+=1
            ii = 0
        
        with open(pathing + '/Vals_hybrid.npy', 'wb') as f:
            np.savez(f, Fsol3_store = Fsol3_store[:,:,k], x3_store = x3_store[:,:,k])

        k+=1



print('############################  loading plot data  ########################################')
Fsol3_store = np.zeros((iter_max*multi,max_try,num_dims ))
x3_store = np.zeros(((iter_max*multi,max_try ,num_dims)))
k = 0
for ndim in dim_list:
    pathing = path_front  + str(ndim)#date_time
    filename_hybrid = pathing + '/Vals_hybrid.npy'
    f_hyb = np.load(filename_hybrid)
    x3_store[:,:,k] = f_hyb['x3_store']
    Fsol3_store[:,:,k] = f_hyb['Fsol3_store']
    k+=1


print('############################  plotting now  ########################################')
if plots_vals:
    
    ### plots the number of function evaluation vs the function value
    fig4,ax4 = plt.subplots(nrows = 2, sharex = True)
    x3_store_list = []
    x3_store_0_list = []
    Fsol3_store_list = []
    variance3_list = []
    variance_s3_list = []
    labels = []
    handles = []
    for k in range(num_dims):
        print('doing computations for dimension %i',k)

        x3_store_0 = np.average(x3_store[:,:,k], axis = 1)[0]
        x3_store_0_list.append(x3_store_0)
        ss = np.argmax(Fsol3_store[-1,:,k])
        if int(x3_store_0) == 1:
            x3_store_list.append(np.append( [1], np.average(x3_store[:,:,k], axis = 1)))
            Fsol3_store_list.append(np.append( np.average(Fsol3_store[:,:,k], axis = 1)[0] * np.ones((int(x3_store_0_list[k]),1)), np.average(Fsol3_store[:,:,k], axis = 1)) )

            variance3 = abs(np.average(Fsol3_store[:,:,k]**2, axis = 1) - np.average(Fsol3_store[:,:,k], axis = 1)**2)
            variance3_list.append( np.append( variance3[0] * np.ones((int(x3_store_0_list[k]),1)), variance3) )

        else:
            x3_store_list.append( np.append( range(1,int(x3_store_0_list[k])), np.average(x3_store[:,:,k], axis = 1)) )
            Fsol3_store_list.append( np.append( np.average(Fsol3_store[:,:,k], axis = 1)[0] * np.ones((int(x3_store_0_list[k])-1,1)), np.average(Fsol3_store[:,:,k], axis = 1)) )
            
            variance3 = abs(np.average(Fsol3_store[:,:,k]**2, axis = 1) - np.average(Fsol3_store[:,:,k], axis = 1)**2)
            variance3_list.append( np.append( variance3[0] * np.ones((int(x3_store_0_list[k])-1,1)), variance3) )

    for k in range(num_dims):
        line, = ax4[-1].plot(x3_store_list[k], Fsol3_store_list[k], label = f'dim = {dim_list[k]}')
        ax4[0].plot(x3_store_list[k], np.sqrt(variance3_list[k]), label = f'dim = {dim_list[k]}')

        iterations_bound = 1000

        ax4[0].set_ylabel("Standard Deviation", fontsize = 12)

        ax4[-1].set_xlabel("Function Evaluations", fontsize = 12)
        ax4[-1].set_xscale("log")

        ax4[-1].set_xlim([1,4*10**4])
        ax4[-1].set_ylabel("Function Value")

        ax4[0].set_yscale("symlog", linthresh=1e-3)
        ax4[1].set_yscale("symlog", linthresh=1e-3)


        m = 15### set to number of directional samples
        ax4[0].set_title(r'%i samples per evaluation near discontinuity'%(m))
        fig4.suptitle('Mean Objective Value for %i random runs' %max_try)

        ax4[0].legend(loc='center left', bbox_to_anchor=(1, 0))
        ax4[0].tick_params(axis='y', which='both', left=True)

        plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave right margin free
        plt.subplots_adjust(hspace=0.25)

        iterations_bound = 1000

        ax3[0].set_ylabel("Standard Deviation", fontsize = 12)

        ax3[-1].set_xlabel("Function Evaluations", fontsize = 12)
        ax3[-1].set_xscale("log")
        ax3[-1].set_xlim([1,4*10**4])
        ax3[-1].set_ylabel(r"$\mathbf{x}_1$")


        m = 10### set to number of directional samples
        fig3.suptitle(r'Mean $\mathbf{x}_1$ Value for %i random runs' %max_try)
plt.subplots_adjust(hspace=0.2)


print('Final Function Values:')
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], Fsol3_store_list[k][-1] ) )
print('Final Standard Deviation Values:' )
for k in range(num_dims):
    print(' dim %i: %f' %( dim_list[k], np.sqrt(variance3_list[k])[-1] ) )

val_av = np.zeros(num_dims)
print('Weighted Average Cost of an Optimization Step until Convergence:')
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    val_av[k] = np.sum(x3_store[conv3_b,range(len(conv3_b))])/np.sum(conv3_b)
    print(' dim %i: %f' %( dim_list[k], val_av[k] ) )

val_tot = np.zeros(num_dims)
print('Number of Function Evaluations until Convergence:' )
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    val_tot[k] = np.average(x3_store[conv3_b,range(len(conv3_b))])
    print(' dim %i: %f' %( dim_list[k], val_tot[k] ) )
print('Number of Steps until Convergence:' )
for k in range(num_dims):
    conv3_b = np.argmax(Fsol3_store[:,:,k] == Fsol3_store[-1,:,k], axis = 0)
    conv3_c = int(np.average(conv3_b))
    print(' dim %i: %f' %( dim_list[k], np.average(conv3_b) ) )



############################## MLE estimation #######################
def power_model(x,a,b,c):
    return a + b * np.power(x,c)

p0_tot =  [400,100,0.2]
p0_av =  [329, 128, 0.02]
(popt_tot, pcov_tot) = curve_fit(power_model, dim_list, val_tot, p0 = p0_tot)
(popt_av, pcov_av) = curve_fit(power_model, dim_list, val_av ,p0 = p0_av)
perr_av = np.sqrt(np.diag(pcov_av))
perr_tot = np.sqrt(np.diag(pcov_tot))
print('parameters tot:', popt_tot)
print('parameters av:' ,popt_av)
print('CI tot:', 1.96 * perr_tot)
print('CI av:' , 1.96 * perr_av)
xxx = np.linspace(dim_list[0], dim_list[-1], 1000)
power_model_tot = lambda x: power_model(x, popt_tot[0], popt_tot[1],popt_tot[2])
power_model_av = lambda x: power_model(x, popt_av[0], popt_av[1],popt_av[2])
p30 = power_model_tot(xxx)
p3 = power_model_av(xxx)

resid_av = np.zeros(len(dim_list))
resid_tot = np.zeros(len(dim_list))

for k in range(len(dim_list)):
    resid_tot[k] = val_tot[k]  - power_model_tot(dim_list[k])
    resid_av[k]= val_av[k]  - power_model_av(dim_list[k])

rmse_tot = np.sqrt(np.mean(resid_tot**2))
ss_res_tot = np.sum(resid_tot**2)
ss_t_tot = np.sum((val_tot - np.mean(val_tot))**2)
r_sq_tot = 1 - ss_res_tot/ss_t_tot

print('RMSE tot:', rmse_tot)
print('R^2 tot:', r_sq_tot)


rmse_av = np.sqrt(np.mean(resid_av**2))
ss_res_av = np.sum(resid_av**2)
ss_t_av = np.sum((val_av - np.mean(val_av))**2)
r_sq_av = 1 - ss_res_av/ss_t_av

print('RMSE av:', rmse_av)
print('R^2 av:', r_sq_av)


figp, axp = plt.subplots(nrows = 2)
axp[0].scatter(dim_list, val_tot, label = "Samples")#label = r'$c_\mathrm{mean}$')
axp[0].plot(xxx, p30, ls = '--',  label = r'$p_\mathrm{tot}$')
axp[1].scatter(dim_list, val_av, label = "Samples")#label = r'$c_\mathrm{mean}^\mathrm{step}$')
axp[1].plot(xxx, p3, ls = '--', label = r'$p_\mathrm{step}$')
axp[1].set_xlabel(r"Dimension $N$", fontsize = 12)
axp[1].set_xscale("log")
axp[0].set_xscale("log")
axp[1].set_yscale("log")
axp[0].set_yscale("log")
axp[1].set_ylabel(r'$c_\mathrm{mean}^\mathrm{step}$', rotation=0, fontsize = 14, labelpad=20)
axp[0].set_ylabel(r'$c_\mathrm{mean}$', rotation=0, fontsize = 14, labelpad=20)
axp[1].set_xlim([dim_list[0],dim_list[-1]])
axp[0].set_xlim([dim_list[0],dim_list[-1]])
axp[0].legend(loc = 'upper left', fontsize = 12)
axp[1].legend(loc = 'upper left', fontsize = 12)
figp.suptitle('Number of mean total cost and mean cost per step')
plt.subplots_adjust(hspace=0.2)



plt.show()


















