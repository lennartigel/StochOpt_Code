# ASM vs GSM Optimization Experiments Data

This repository contains simulation data to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)**. 

Data is structured into two folders, 'Ex_nGs_final_circ_exrun_new' and 'Ex_nGs_final_circ_new'. 

--------------------------------------------------------

Folder 'Ex_nGs_final_circ_new' contains the data for the random simulations done. 
This data was generated with the script 'gd_ex_nGs.npy' 
Its data has the stucture: 

metadata.npy -- stores run metadata, 
'max_try' = number of runs
'iter_max' = maximum number of iterations for GSM
'multi' = multiplicator of iter_max to get maximum number of iterations for ASM

Run_X_results1.npy --- stores results for optimization run for GSM; X = run number
'x1' = Vector of length 'iter_max' storing number of function evaluations per iteration step
'sol1' = Vector of shape 'iter_max' x 2 storing iterate for each iteration step

Run_X_results3.npy --- stores results for optimization run for AM; X = run number
'x3' = Vector of length 'iter_max'*'multi' storing number of function evaluations per iteration step
'sol3' = Vector of shape 'iter_max'*'multi' x 2 storing iterate for each iteration step

Vals_stoch.npy --- assembles function values and function evaluations values for GSM
'Fsol1_store' = assembly vector of length 'iter_max' of evaluated GSM function values at each iterate in 'sol1'
'x1_store' = assembly vector of shape 'iter_max' x 'max_try' of all 'x1' values

Vals_hyb.npy --- assembles function values and function evaluations values for ASM
'Fsol3_store' = assembly vector of length 'iter_max'*'multi' of evaluated ASM function values at each iterate in 'sol3'
'x3_store' = assembly vector of shape 'iter_max'*'multi' x 'max_try' of all 'x3' values

#### this assembly data is read out with the script 'load_runs_ex_nGs.npy'

--------------------------------------------------------

Folder 'Ex_nGs_final_circ_exrun_new' contains the data for the example run. 
Its data has the stucture:

metadata.npy -- stores run metadata, 
'max_try' = number of runs
'iter_max' = maximum number of iterations for GSM
'multi' = multiplicator of iter_max to get maximum number of iterations for ASM

Run_X_results1.npy --- stores results for optimization run for GSM; X = run number
'x1' = Vector of length 'iter_max' storing number of function evaluations per iteration step
'sol1' = Vector of shape 'iter_max' x 2 storing iterate for each iteration step

Run_X_results3.npy --- stores results for optimization run for AM; X = run number
'x3' = Vector of length 'iter_max'*'multi' storing number of function evaluations per iteration step
'sol3' = Vector of shape 'iter_max'*'multi' x 2 storing iterate for each iteration step

#### No assembly was done for this example run, as its only used to plot data with the script 'load_landscape_ex_nGs.npy'
