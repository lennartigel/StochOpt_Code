# ASM vs GSM Optimization Experiments Data

This repository contains simulation data to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)**. 

Data is structured into one folder 'comp_big_new' with multiple subfolders. 
Each subfolder has the name structure: 'final_2_to_17_new_N' where N corresponds to dimension
N = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
--------------------------------------------------------

Folder 'final_2_to_17_new_N' contains the data for the random simulations done. 
This data was generated with the script 'gd_ex2_100d_hpc.npy' on the woody cluster of the HPC of the FAU Erlangen-Nürnberg
Its data has the stucture: 

metadata.npy -- stores run metadata, 
'max_try' = number of runs
'iter_max' = maximum number of iterations for GSM
'multi' = multiplicator of iter_max to get maximum number of iterations for ASM (HERE 1)
'sigma' = sigma size chosen in 'gd_ex2_100d.npy'

Run_X_results1.npy --- stores results for optimization run for GSM; X = run number
'x1' = Vector of length 'iter_max' storing number of function evaluations per iteration step
'sol1' = Vector of shape 'iter_max' x N storing iterate for each iteration step

Run_X_results3.npy --- stores results for optimization run for ASM; X = run number
'x3' = Vector of length 'iter_max'*'multi' storing number of function evaluations per iteration step
'sol3' = Vector of shape 'iter_max'*'multi' x N storing iterate for each iteration step

Vals_stoch.npy --- assembles function values and function evaluations values for GSM
'Fsol1_store' = assembly vector of length 'iter_max' of evaluated GSM function values at each iterate in 'sol1'
'x1_store' = assembly vector of shape 'iter_max' x 'max_try' of all 'x1' values

Vals_hyb.npy --- assembles function values and function evaluations values for ASM
'Fsol3_store' = assembly vector of length 'iter_max'*'multi' of evaluated ASM function values at each iterate in 'sol3'
'x3_store' = assembly vector of shape 'iter_max'*'multi' x 'max_try' of all 'x3' values

#### this assembly data is generated with the script 'load_runs_ex2_100d_hpc.npy' on the woody cluster of the HPC of the FAU Erlangen-Nürnberg
#### it is visualized by running the script 'load_runs_ex2_100d_hpc.npy'