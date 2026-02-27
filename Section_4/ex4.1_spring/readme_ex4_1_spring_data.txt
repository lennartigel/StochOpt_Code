# ASM vs GSM Optimization Experiments Data

This repository contains simulation data to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)**. 
As well as generated landscape data. 

Data is structured into two folders, 'ExAS_final_new' and 'LS'. 

--------------------------------------------------------

Folder 'ExAS_final_new' contains the data for the random simulations done. 
This data was generated with the script 'gd_ex_spring_hpc_new.py' 
Its data has the stucture: 

metadata.npy -- stores run metadata, 
'max_try' = number of runs
'iter_max' = maximum number of iterations for GSM
'multi' = multiplicator of iter_max to get maximum number of iterations for ASM
'sigma' = chosen value for sigma in optimization


Run_X_results_stoch.npy --- stores results for optimization run for GSM; X = run number
'x1' = Vector of length 'iter_max' storing number of function evaluations per iteration step
'sol1' = Vector of shape 'iter_max' x 2 storing iterate for each iteration step

Run_X_results_hyb.npy --- stores results for optimization run for ASM; X = run number
'x3' = Vector of length 'iter_max'*'multi' storing number of function evaluations per iteration step
'sol3' = Vector of shape 'iter_max'*'multi' x 2 storing iterate for each iteration step
'nu3' = Vector of shape 'iter_max'*'multi' x 2 storing smoothing direction for each iteration step
'eps_val' = Scalar value of chosen epsilon value for ASM
'kappa' = Scalar value of chosen kappa value for ASM

Run_XVals_stoch.npy --- stores function values for each optimization run of GSM; X = run_number
'Fsol1_store' = Vector of shape 'iter_max' x 'max_try' with function values of GSM at K-th iterate of X-th run stored in entry [K,X] (unfortunately otherwise zero entries in matrix due to my oversight)
'x1_store' = Vector of shape 'iter_max' x 'max_try' with 'x1_store'[:,X] = 'x1' for X-th run, otherwise 0 (due to oversight). 

Run_XVals_hyb.npy --- stores function values for each optimization run of ASM; X = run_number
'Fsol3_store' = Vector of shape 'iter_max'*'multi' x 'max_try' with function values of ASM at K-th iterate of X-th run stored in entry [K,X] (unfortunately otherwise zero entries in matrix due to my oversight)
'x3_store' = Vector of shape 'iter_max'*'multi' x 'max_try' with 'x3_store'[:,X] = 'x3' for X-th run, otherwise 0 (due to oversight). 

#### this data is generated with the script 'load_runs_ex_spring_hpc_new.py'

Vals_stoch.npy --- assembles function values and function evaluations values for GSM
'Fsol1_store' = assembly vector of length 'iter_max' of evaluated GSM function values at each iterate in 'sol1'
'x1_store' = assembly vector of shape 'iter_max' x 'max_try' of all 'x1' values

Vals_hyb.npy --- assembles function values and function evaluations values for ASM
'Fsol3_store' = assembly vector of length 'iter_max'*'multi' of evaluated ASM function values at each iterate in 'sol3'
'x3_store' = assembly vector of shape 'iter_max'*'multi' x 'max_try' of all 'x3' values

#### this assembly data is assembled and read out with the script 'plot_runs_ex_spring_new.py'

--------------------------------------------------------

Folder 'LS' contains the landscape data.
The landscape is generated in seperate processes, stored in assembly files from which then a full landscape file is assembled. 
The assembly files contain:

final_springX_Glob_Landscape.npy --- stores landscape data generated for GSM
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'vals_stoch' = list of function evaluations of GSM at each (x[1],x[2]) coordinate assigned to the X-th process.

final_springX_Adap_Landscape.npy --- stores landscape data generated for ASM
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'vals_adap' = list of function evaluations of ASM at each (x[1],x[2]) coordinate assigned to the X-th process.
'grads' = list of smoothing direction (when smoothing is performed) at given (x[1],x[2]) coordinate assigned to the X-th process.
'eps_val' = chosen epsilon value for ASM 
'kappa' = chosen kappa value for ASM

final_springX_Regular_Landscape.npy --- stores landscape data generated for f and G
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'vals' = list of function evaluations of f at each (x[1],x[2]) coordinate assigned to the X-th process.
'eigs' = list of function evaluations of G(in this case lambda_1) at each (x[1],x[2]) coordinate assigned to the X-th process.

#### this partial landscape data was generated with the script 'load_landscape_ex_spring_hpc_new.py'

### the assembled landscape data is found in

final_spring_Smooth_Landscape.npy --- stores assembled landscape data for GSM, ASM, f and GSM
'vals' = list of function evaluations of f at each (x[1],x[2]) coordinate
'vals_g' = list of function evaluations of GSM at each (x[1],x[2]) coordinate
'vals_ad' = list of function evaluations of ASM at each (x[1],x[2]) coordinate
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'E' = list of function evaluations of G(in this case lambda_1) at each (x[1],x[2]) coordinate

#### this assembled landscape data is assembled and read out with the script 'plot_ls_ex_spring_new.py'
