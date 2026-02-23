# ASM vs GSM Optimization Experiments Data

This repository contains simulation data to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)**. 
As well as generated landscape data. 

Data is structured into two folders, 'Ex2_final_exrun_new' and 'Ex2_final_new'. 

--------------------------------------------------------

Folder 'Ex2_final_new' contains the data for the random simulations done. 
This data was generated with the script 'gd_ex2_new.npy' 
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

#### this assembly data is read out with the script 'load_runs_ex2_new.npy'

--------------------------------------------------------

Folder 'Ex2_final_exrun_new' contains the data for the example run. 
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
#### No assembly was done for this example run, as its only used to plot data with the script 'load_landscape_ex2_new.npy'

Landscape_stoch.npy --- stores landscape data generated for GSM
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'z' = matrix of function evaluations of GSM at each (x[1],x[2]) coordinate
'minz' = minimal value in GSM landscape
'min_coords' = coordinates where minimal value is assumed

Landscape_hyb.npy --- stores landscape data generated for ASM
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'z' = matrix of function evaluations of ASM at each (x[1],x[2]) coordinate
'minz' = minimal value in ASM landscape
'min_coords' = coordinates where minimal value is assumed

Landscape_exact.npy --- stores landscape data generated for f
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'z' = matrix of function evaluations of f at each (x[1],x[2]) coordinate
'minz' = minimal value in f landscape

#### landscape data was generated and visualized with the script 'load_landscape_ex2_new.npy'