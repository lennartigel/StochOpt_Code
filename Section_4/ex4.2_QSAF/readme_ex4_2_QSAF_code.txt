# ASM Optimization Experiments QSAF

This repository contains simulation code to apply  the **Adaptive Smoothing Method (ASM)** to the QSAF problem. 

Most of this code is written with a wrapper function to be called for a given process and must thus be called with "python -c 'import filename as bsp; bsp.run_fun(X)'" where X denotes what id the process has. 
We denote files that have to be run this way by: filename.py -*hpc*-

## 1. Springclass2D_hpc.py and Springclass2D_hpc_backstepping.py
Performs simulation of QSAF model for different setups as specified in the file and different material strengths. 
class object is defined by calling 'SpringFracture' with the following inputs: 
### SpringFracture parameters
| Parameter | Description |
|-----------|-------------|
| L | number of nodes in height/length |
| B | number of nodes in width |
| n_time_steps | number of timesteps to perform |
| ntnn | specifies whether to perform simulation with only nearest neighbor bonds or also with next to nearest neighbors | 
| end_time | final simulation time |
| ref_fact | refinement factor at time of fracture (only for "Springclass2D_hpc_backstepping.py")
Afterwards you can manipulate the following parameters. 
### SpringFracture class  parameters 
| Parameter | Description |
|-----------|-------------|
| boundary | list of boundaries on which boundary conditions are applied for specified boudnary; codes include 'upper' (entire upper boundary); 'lower' (entire lower boundary); 'upper_left' (left half of upper boundary); 'lower_left'(left half of lower boundary); 'upper_right'(right half of upper boundary); 'lower_right'(right half of lower boundary); 'left'(entire left boundary); 'right'(entire right boundary); 'right_upper'(top half of right boundary); 'right_lower'(lower half of right boundary); 'left_upper'(top half of left boundary); 'left_lower'(lower half of left boundary) |
| loading | list of boundary condition functions for each entry in 'boundary' (specify loading_spatial_dep = True if they depend on space as well as time, False otherwise), should have form [r0,r1] for each entry in 'boundary'| 
| dim | spcifies in which dimension each subentry of the list for a boundary condition aplies to; [0,1] means first entry r0 applies in x, second entry r1 in y direction | 
| inhom_material | Boolean that specifies whether homogenous setup or inhomogenous setup (we generally set this to True in all scripts)|
| compute_mat_der | Boolean on whether design derivatives should be computed |
| id | specifies internal id string of simulaiton; for reference wen storing data |
| track_eigs | Boolean that specifies whether the eigenvalues should be track in each timestep |
| chevron; topdown; leftright; circles; Friedrich_setup; Friedrich_setup_2 | Booleans that specifies what type of inhomogeniety is introduced and what design dependency |
| num_alpha | number of design variables for internal reference |
| alpha_list | list of design variable values so they can be applied to material |
| alpha_fix_list | list of material parameter values so they can be applied during initlaization and optimization to inhomogeniety; Set this to your desired material | 

Additionally you should set these parameters. 
### SpringFracture setup parameters 
| Parameter | Description |
|-----------|-------------|
| loading_spatial_dep| Boolean that tells the method, that your boundary conditions also depend on space and on time, insted of only on time | 
| diss_type | dissipation type; available types are "L2" (default) and "KV" (not entirely implemented) |
| nu | dissipation value |
| min_eig | minimal value at which its assumed time of fracture occurs when lambda0 goes below it (only relevant for "Springclass2D_hpc_backstepping.py") |
| num_eigs | number of eigenvalues to compute (x computes the smallest x eigenvalues) | 
| save_output | Boolean that tells the method to store the output or not |
| print_progress | Boolean that tells the method to print its progress to the console while running | 



To run the simulation, call SpringFracture.solve with the specified parameter
| Parameter | Description |
|-----------|-------------|
| method | solution method to solve the energy minimization problem.  only 'one step' is available |

It generates the following output called by SpringFracture.variablename
| output | Description |
|-----------|-------------|
| Y | solution vector of nodal position for each timestep |
| Y_alpha | design derivate of solution vector at nodal position for each timestep | 
| energies | list of energy values at each timestep | 
| objectives | list of objective( energy plus dissipation) values at each timestep |
| hist_list | vector of history variable for nn bonds at each timestep |
| hist_ntnn_list | vector of history variable for ntnn bonds at each timestep |
| hist_list_alpha | vector of design derivative of history variable for nn bonds at each timestep |
| hist_ntnn_list_alpha | vector of design derivative of history variable for ntnn bond at each timestep |  

The following output is stored in "meta.npy", if specified. 
| output | Description |
|-----------|-------------|
| ntnn | Boolean of whether its an ntnn or nn simulation | 
| L | value of L | 
| B | value of B |
| max_steps | n_time_steps |

The following output is stored in "data.npy", if specified. 
| output | Description |
|-----------|-------------|
| x | solution vector of nodal position for each timestep |
| eigs | vector of smallest eigenvalue at each timestep |
| eigsn | vector of n smallest eigenvalues at each timestep |
| T | list of timesteps | 
| energy | list of energy values at each timestep | 
| objective | list of objective( energy plus dissipation) values at each timestep |
| r_hist | vector of history variable for nn bonds at each timestep |
| R_1 | vector of R_1 value for each nn bond | 
| R_2 | vector of R_2 value for each nn bond | 
| r_hist_ntnn | vector of history variable for ntnn bond at each timestep |  
| R_1_ntnn | vector of R_1 value for each ntnn bond | 
| R_2_ntnn | vector of R_2 value for each ntnn bond | 

Additionally, if inhom_material= true and compute_mat_der = true, then the following output is stored in "data.npy", if specified. 

| output | Description |
|-----------|-------------|
| x_alpha | design derivatives of solution vector of nodal position for each timestep |
| energy_mat | list of design derivatives of energy values at each timestep | 
| objective_mat | list of design derivatives of objective( energy plus dissipation) values at each timestep |
| r_hist_alpha | vector of design derivatives of history variable for nn bonds at each timestep |
| r_hist_ntnn_alpha | vector of design derivatives of history variable for ntnn bond at each timestep |  
| J_list | list of jacobian of objective at each timestep |
| J_list_alpha | list of design derivatives of jacobian of objective at each timestep |

---

## 2. gd_exMS_hpc.py -*hpc*-
Performs optimization runs from random starting points in the domain "[0.1,2]^2" on an objective with a spring problem as underlying state problem to be solved. 
The state problem is solved by the script in Springclass_hpc_AS.py . 
Includes the class GradientDescent, which implements the ASM or GSM depending on input. 
For construction it requires the following parameters:
### GradientDescent Parameters
| Parameter | Description |
|-----------|-------------|
| xstart | Starting point for optimization |
| npop | Number of samples for GSM quadrature; affects ASM only if "use_one_directional_smoothing=False" |
| sigma | Maximum length of the line used for smoothing if ASM OR diameter of smoothing set if GSM|
| check_for_stability | If True, ASM will be used |
| use_one_directional_smoothing | True = smooth along line; False = full local volume smoothing |
| gradient_available | True if a reliable gradient exists; False triggers smoothing |
| npop_der | ASM only; number of derivative evaluations at points where gradient exists (set to 1) |
| npop_stoch | ASM only; number of evaluations for line smoothing |
| ident | identifier of the run, used to identify run in print outputs | 

### The following parameters are internally set
| Parameter | Description |
|-----------|-------------|
| tol | Small offset to avoid instability of gradient on smoothing set boundary |

~~~ Armijo Parameters
| Parameter | Description |
|-----------|-------------|
| theta | Fixed at 0.001 |
| aleph | Initial search length |
| omega | Shrinking factor fixed at 0.5 |

###Optimization is performed by calling the optimization function of GradientDescent with the following inputs.
| Input | Description |
|-----------|-------------|
| fmin | Function that returns for each input the value of f, if called with only one argument; values (f, df) if called with two inputs; |
| fminE | Function that returns for each input the value of G, if called with only one argument; values (G, dG) if called with two inputs; |
| gradient_function | ### deprecated;  set to NONE
| iterations | number of iterations |
| epsilon | List of widths of belt around discontinuity for smoothing for each level set function; if only one entry, then assumed to be same for all |
| kappa | List of widths of interior belt with max smoothing line length for each level set function; if only one entry, then assumed to be same for all |
| aleph | Initial search length for Armijo |
| id | helps identify the run and tells it where to store savepoint data |

## NOTE: epsilon AND kappa ARE SQUARED INTERNALLY, SO HAND OVER sqrt(epsilon) and sqrt(kappa)

### Output

| Variable | Notes |
|----------|------|
| sol | Array of iterates for a given run |
| x | Count of function evaluations per iteration for a given run |
| nu | Array of smoothing directions for each iterate for a given run |

---

## 3. load_runs_MS_hpc_new.py -*hpc*-

- Computes function values at iterates for ASM.
- processes output data from "gd_exMS_hpc.py" and stores the following output in "Run_XVals_hybrid.npy" and "Run_XVals_stoch.npy" for X-th run of ASM, respectively

| Variable | Notes | 
|---------|-------|
| Fsol_store | Array of values of either ASM  at each iterate for run X | 
| x_store | Array of x function evaluations count for each iterate for run X |

---

## 4. plot_runs_MS_hpc_new.py 

- Calculates mean, standard deviation, and prints values of "c_mean", "k_mean", and "c_mean^step".
- Plots the mean and standard deviation over optimization iterations
- **do_work**: Set to True to assemble output data from "load_runs_MS_hpc_new.py" and store the following output in "Vals_hybrid.npy" and "Vals_stoch.npy" for ASM , respectively

| Variable | Notes | 
|---------|-------|
| Fsol_store | Array of assembled values of either AS at each iterate for each run | 
| x_store | Array of assembled function evaluations count for each iterate for each run |

---
## 5. load_landscape_MS_hpc_ASM.py -*hpc*-

Generates landscapes for ASM
For process number X stores the following in  "codeX_Adap_Landscape" + suffix + ".npy" for f and ASM,respectively.
**code** and **suffix** are internally set string parameter that assigns an id to the filename

codeX_Adap_Landscape.npy --- stores landscape data generated for ASM for process X
| Variable | Notes | 
|---------|-------|
| x_1 | matrix of x[1] meshgrid values 
| x_2 | matrix of x[2] meshgrid values 
| vals_adap | list of function evaluations of ASM at each (x[1],x[2]) coordinate assigned to the X-th process.
| eps_val |  chosen epsilon value for ASM 
| kappa |  chosen kappa value for ASM

---

## 6. assemble_ls_MS_ASM.py
- Plots and assembles the landscapes for ASM
- **do_work**: Set to True to assemble output data from "load_landscape_MS_hpc_ASM.py" and store the following output in "Complete_MS_Adap_Landscape" + suffix + ".npy" for f and ASM.

| Variable | Notes | 
|---------|-------|
| vals |  list of function evaluations of ASM at each (x[1],x[2]) coordinate
| x_1 |  matrix of x[1] meshgrid values 
| x_2 |  matrix of x[2] meshgrid values 

---


## 7. load_landscape_MS_hpc_dd.py -*hpc*-

Generates landscapes for f, energy, eigenvalues of QSAF problem at each design variable in [0.1,2]^2 or [0.1,4]^2 with a refinement around the diagonal to accurately capture the discontinuity. 
Additionally, the file used for the QSAF simulation "Springclass2D_hpc_backstepping.py" has an additional method to allow refinement in the timeframe of fracture to better capture the minimal eigenvalue.
For process number X stores the following in "codeX_Regular_Landscape.npy" and "codeX_Adap_Landscape.npy" for f and ASM,respectively.
**code** is an internally set parameter that assigns an id to the filename

codeX_Spring_Landscape.npy --- stores landscape data generated for ASM for process X
| Variable | Notes | 
|---------|-------|
| eigs | minimal value for all 4 smallest eigenvalues over the time evolution of the QSAF model at design(x[1], x[2]) |
| eigs_2 | ### deprecated secondary list of eigenvalue minima 
| eigs_extra | 4 x number of timesteps sized vector of the evolution of four smallest eigenvalues over time for QSAF simulation at design(x[1], x[2]) |
| energy | energy at final time T of the QSAF model at design(x[1], x[2]) |
| energy_og | energy at starting time 0 of the QSAF model at design(x[1], x[2]) |
| energy_diff | energy difference between 'energy' and 'energy_og' |
| energy_mat_0 | derivative in x[1] direction for 'energy' |
| energy_mat_1 | derivative in x[2] direction for 'energy' |
| energy_og_mat_0 | derivative in x[1] direction for 'energy_og' |
| energy_og_mat_1 | derivative in x[2] direction for 'energy_og' |
| energy_diff_mat_0 | derivative in x[1] direction for 'energy_diff'  |
| energy_diff_mat_1 | derivative in x[2] direction for 'energy_diff' |
| x_1 | list of x[1] values for evaluation points in the landscape  |
| x_2 | list of x[2] values for evaluation points in the landscape |

---

## 8. assemble_ls_MS_hpc_dd.py
- Plots the landscapes for ASM, and the discontinuous objective function from assembled landscape files
- **do_work**: Set to True to assemble output data from "load_landscape_MS_hpc_dd.py" and store the following output in "code_Spring_Landscape.npy" for f and ASM.

| Variable | Notes | 
|---------|-------|
| eigs |  vector of minimal eigenvalue Lambda_0 over the time evolution for each design (x[1], x[2]) |
| eigs1 |  vector of eigenvalue Lambda_1 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2])  |
| eigs2 |  vector of eigenvalue Lambda_2 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2])   |
| eigs3 |  vector of eigenvalue Lambda_3 evaluated at time where Lambda_0 takes its minimum over the time evolution for each design (x[1], x[2])  |
| eigs0_normal |  Lambda_0 normalizes w.r.t. norm of design point (x[1], x[2]) |
| eigs1_normal |  Lambda_1 normalizes w.r.t. norm of design point (x[1], x[2]) |
| energy |  energy at final time T of the QSAF model at design(x[1], x[2]) |
| energy_og |  energy at starting time 0 of the QSAF model at design(x[1], x[2]) |
| energy_diff |  energy difference between 'energy' and 'energy_og' |
| energy_mat_0 |  derivative in x[1] direction for 'energy' |
| energy_mat_1 |  derivative in x[2] direction for 'energy' |
| energy_og_mat_0 |  derivative in x[1] direction for 'energy_og' |
| energy_og_mat_1 |  derivative in x[2] direction for 'energy_og' |
| energy_diff_mat_0 |  derivative in x[1] direction for 'energy_diff'  |
| energy_diff_mat_1 |  derivative in x[2] direction for 'energy_diff' |
| x_1 |  list of x[1] values for evaluation points in the landscape  |
| x_2 |  list of x[2] values for evaluation points in the landscape |

---

9. gen_MS_sims.py -*hpc*-
- runs a QSAF simulation at specified designs with refinement around the time of fracture and stores the simulation data in specified folder using "Springclass2D_hpc_backstepping.py".

---

10. plot_MS_sims.py 
- loads the data stored by "gen_MS_sims.py" and plots the eigenvalue evolution and fracture evolution over time in a specified folder destination using the script in "Springclass2D_plot.py". 

---

11. Springclass2D_plot
- used to plot crack evolution from data files, refer to "plot_MS_sims.py" for how to specify the setup to plot crack evolution to a given directory.


## How to Run simulations: 

1. Adjust "Spring_fct"(f,df) and "Eig_fct"(G) in "gd_exMS_hpc.py" to set your objective and levelset function.
2. Set parameters in "gd_exMS_hpc.py", including metadata and state problem parameters.
3. Run "gd_exMS_hpc.py".
4. Use "load_runs_MS_hpc_new.py" to compute values for each run.
5. Use "plot_runs_MS_hpc_new.py" to assemble values and generate plots for specified dataset.

6a. Use "load_landscape_MS_hpc_dd.py" to generate landscape files of f and eigenvalues for specified dataset, set parameters in file.
6b. Use "assemble_ls_MS_hpc_dd.py" to assemble and visualize landscape files of f and eigenvalues for specified dataset.

7a. Use "load_landscape_MS_hpc_ASM.py" to generate landscape files of ASM for specified dataset, set parameters in file.
7b. Use "assemble_ls_MS_ASM.py" to assemble and visualize landscape files of ASM for specified dataset.

8a. Use "gen_MS_sims.py" to simulate crack evolution, set parameters in file.
8b. Use "plot_MS_sims.py" to visualize crack evolution for specified QSAF simulations.


The default folder in "gd_ex_nGs.py" and any methods that generates data is set to 'test', so nothing is overwritten by accident. 
The default value of "do_work=False" in any file using that Boolean for the same reason.
