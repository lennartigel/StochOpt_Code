# ASM vs GSM Optimization Experiments Data

This repository contains simulation data to apply the **Adaptive Smoothing Method (ASM)** to the QSAF problem. 
As well as generated landscape data. 

Data is structured into four folders, 'Ex_QSAF_final', 'LS_ASM', 'LS_f' and '2dsprings' which contains 'QSAF_cracks'. 

--------------------------------------------------------

Folder 'Ex_QSAF_final' contains the data for the random simulations done. 
This data was generated with the script 'gd_exMS_hpc.py' 
Its data has the stucture: 

metadata.npy -- stores run metadata, 
'max_try' = number of runs
'iter_max' = maximum number of iterations for GSM
'multi' = multiplicator of iter_max to get maximum number of iterations for ASM
'sigma' = chosen value for sigma in optimization

Run_X_results_hyb.npy --- stores results for optimization run for ASM; X = run number
'x3' = Vector of length 'iter_max'*'multi' storing number of function evaluations per iteration step
'sol3' = Vector of shape 'iter_max'*'multi' x 2 storing iterate for each iteration step
'nu3' = Vector of shape 'iter_max'*'multi' x 2 storing smoothing direction for each iteration step

Run_XVals_hyb.npy --- stores function values for each optimization run of ASM; X = run_number
'Fsol3_store' = Vector of shape 'iter_max'*'multi' x 'max_try' with function values of ASM at K-th iterate of X-th run stored in entry [K,X] (unfortunately otherwise zero entries in matrix due to my oversight)
'x3_store' = Vector of shape 'iter_max'*'multi' x 'max_try' with 'x3_store'[:,X] = 'x3' for X-th run, otherwise 0 (due to oversight). 

#### this data is generated with the script 'load_runs_ex_MS_hpc_new.py'

Vals_hyb.npy --- assembles function values and function evaluations values for ASM
'Fsol3_store' = assembly vector of length 'iter_max'*'multi' of evaluated ASM function values at each iterate in 'sol3'
'x3_store' = assembly vector of shape 'iter_max'*'multi' x 'max_try' of all 'x3' values

#### this assembly data is assembled and read out with the script 'plot_runs_ex_MS_hpc_new.py'

--------------------------------------------------------

Folder 'LS_ASM' contains the landscape data for the generated landscape for the ASM.
The landscape is generated in seperate processes, stored in assembly files from which then a full landscape file is assembled. 
The assembly files contain:

final_MSX_Adap_Landscape_04_02_32x32_MS_25_E_575_x2=175_sq.npy --- stores landscape data generated for ASM by the X-th process
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'vals_adap' = list of function evaluations of ASM at each (x[1],x[2]) coordinate assigned to the X-th process.
'grads' = list of smoothing direction (when smoothing is performed) at given (x[1],x[2]) coordinate assigned to the X-th process.
'eps_val' = chosen epsilon value for ASM 
'kappa' = chosen kappa value for ASM

#### a note, the x2=175 is an oversight and does not correspond to the function used to generate the data
#### the partial landscape data was generated with the script 'load_landscape_MS_hpc_ASM.py'
#### the assembled landscape data is found in

Complete_MS_Adap_Landscape_04_02_32x32_MS_25_E_575_x2=175_sq.npy --- assembled landscape data generated for the ASM from final_MSX_Adap_Landscape_04_02_32x32_MS_25_E_575_x2=175_sq.npy
'x_1' = matrix of x[1] meshgrid values 
'x_2 = matrix of x[2] meshgrid values 
'vals' = list of function evaluations of ASM at each (x[1],x[2]) coordinate.

#### this assembled landscape data is assembled and plotted with the script 'assemble_ls_MS_ASM.py'

--------------------------------------------------------

Folder 'LS_f' contains the landscape data for the generated landscape for the objective f and eigenvalue functions on both [0,2]^2 and [0,4]^2 
The landscape is generated in seperate processes, stored in assembly files from which then a full landscape file is assembled. 
The assembly files contain:

LS_32x32_post_nu_0001_dense_diag_2x2_pull25_o15_f75_s75_refined5_X_Spring_Landscape.npy --- stores landscape data generated for energy and eigenvalues by process X
'eigs' = minimal value for all 4 smallest eigenvalues over the time evolution of the QSAF model at design(x[1], x[2])
'eigs_2' = ### deprecated secondary list of eigenvalue minima
'eigs_extra' = 4 x number of timesteps sized vector of the evolution of four smallest eigenvalues over time for QSAF simulation at design(x[1], x[2])
'energy' = energy at final time T of the QSAF model at design(x[1], x[2])
'energy_og' = energy at starting time 0 of the QSAF model at design(x[1], x[2])
'energy_diff' = energy difference between 'energy' and 'energy_og'
'energy_mat_0' = derivative in x[1] direction for 'energy'
'energy_mat_1' = derivative in x[2] direction for 'energy'
'energy_og_mat_0' = derivative in x[1] direction for 'energy_og'
'energy_og_mat_1' = derivative in x[2] direction for 'energy_og'
'energy_diff_mat_0' = derivative in x[1] direction for 'energy_diff' 
'energy_diff_mat_1' = derivative in x[2] direction for 'energy_diff'
'x_1' = list of x[1] values for evaluation points in the landscape 
'x_2' = list of x[2] values for evaluation points in the landscape

#### the partial landscape data was generated with the script 'load_landscape_MS_hpc_dd.py'
#### the assembled landscape data is found in

LS_32x32_post_nu_0001_dense_diag_2x2_pull25_o15_f75_s75_refined5__Spring_Landscape.npy --- assembled landscape data generated for energy and eigenvalues from LS_32x32_post_nu_0001_dense_diag_2x2_pull25_o15_f75_s75_refined5__Spring_Landscape.npy
'eigs' = vector of minimal eigenvalue Lambda_0 over the time evolution for each design (x[1], x[2])
'eigs1' = vector of eigenvalue Lambda_1 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2]) 
'eigs2' = vector of eigenvalue Lambda_2 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2])  
'eigs3' = vector of eigenvalue Lambda_3 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2]) 
'eigs0_normal' = Lambda_0 normalizes w.r.t. norm of design point (x[1], x[2])
'eigs1_normal' = Lambda_1 normalizes w.r.t. norm of design point (x[1], x[2])
'energy' = energy at final time T of the QSAF model at design(x[1], x[2])
'energy_og' = energy at starting time 0 of the QSAF model at design(x[1], x[2])
'energy_diff' = energy difference between 'energy' and 'energy_og'
'energy_mat_0' = derivative in x[1] direction for 'energy'
'energy_mat_1' = derivative in x[2] direction for 'energy'
'energy_og_mat_0' = derivative in x[1] direction for 'energy_og'
'energy_og_mat_1' = derivative in x[2] direction for 'energy_og'
'energy_diff_mat_0' = derivative in x[1] direction for 'energy_diff' 
'energy_diff_mat_1' = derivative in x[2] direction for 'energy_diff'
'x_1' = list of x[1] values for evaluation points in the landscape 
'x_2' = list of x[2] values for evaluation points in the landscape


#### this assembled landscape data is assembled and read out with the script 'assemble_ls_MS_hpc_dd.py'

--------------------------------------------------------

Folder '2dsprings/QSAF_cracks' contains the data for cracks and eigenvalues of QSAF simulations evaluated over the landscape [0,4]^2
Each subfolder has the form 'EX_post_nu_0001_pull25_o15_f75_s75_nebendiag_ex_alph0_x[1]_alph1_x[2]' where x[1] and x[2] denote the design at which the simulation was evaluated and contains: 

meta.npy --- file of basic simulation parameters
'ntnn = Boolean of whether the simulation was NtNN (True) or simply NN (False)
'L = value of number of nodes in length (y direction)
'B = value of number of nodes in width (x direction)
'max_steps = number of time steps

data.npy --- file of simulation results to recreate system state at given time
'x' = vector of length L * B * 2 x num_time_steps of position for each node at each point in time in the simulation
'eigsn' = vector of n smallest eigenvalues at each point in time(n depends on how many were asked to store)
'eigs' = vector of smallest eigenvalue at each point in time 
'T' = end time of simulation 
'energy' = list of energy values at each point of time in the simulation
'objective' = list of QSAF complete energy fucntion (energy plus dissipation) values at each point of time in the simulation 
'r_hist' = list of history variable for NN bonds values at each time step
'R_1' = value of R_1 for each NN bond
'R_2' = value of R_2 for each NN bond
'r_hist_ntnn' = list of history variable for NtNN bonds values at each time step 
'R_1_ntnn' = value of R_1_ntnn for each NtNN bond
'R_2_ntnn' = value of R_2_ntnn for each NtNN bond

data_der.npy ---
'x_alpha' = design derivative of x vector, size L * B * 2 x num_time_steps x 2 
'energy_mat' = design derivatives of energy 
'objective_mat' = design derivatives of objective
'r_hist_alpha' = design derivative values of history variable for NN bonds 
'r_hist_ntnn_alpha' = design derivative values of history variable for NN bonds 
'J_list' = list of Jacobians wrt to bonds of objectives for each time step (dx objective)
'J_list_alpha' = list of design derivatives of Jacobian of objective for each time step (dalpha dx objective)

#### Note that objective here is NOT the objective of some design optimization but the objecitve of the QSAF energy minimization problem 
#### The data was generated using the script 'gen_MS_sims.py' and is read out/plotted with the script 'plot_MS_sims.py' 