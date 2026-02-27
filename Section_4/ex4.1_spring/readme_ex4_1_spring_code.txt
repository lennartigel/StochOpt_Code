# ASM vs GSM Optimization Experiments

This repository contains simulation code to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)** on application to a spring problem.

Most of this code is written with a wrapper function to be called for a given process and must thus be called with "python -c 'import filename as bsp; bsp.run_fun(X)'" where X denotes what id the process has. 
We denote files that have to be run this way by: filename.py -*hpc*-

## 1. Springclass_hpc_AS.py
Performs simulation of spring system with two interfaces that eahc have a different material strength for specified input. 
class object is defined by calling 'FractureProblem' with the following inputs: 
### FractureProblem parameters
| Parameter | Description |
|-----------|-------------|
| n_time_steps | number of timesteps to perform (without refinement applied) |
| end_time | final simulation time |

Afterwards you can manipulate the material parameters. 
### FractureProblem class material parameters 
| Parameter | Description |
|-----------|-------------|
| G1 | fracture toughness of interface 1, this is the design dependency for x[1] |
| c1 | elasticity constant for interface 1|
| G2 | fracture toughness of interface 2, this is the design dependency for x[2] |
| c2 | elasticity constant for interface 2 |
| k | spring constant |

To run the simulation, call FractureProblem.solve with the specified parameter
| Parameter | Description |
|-----------|-------------|
| method | solution method to solve the energy minimization problem.  default is 'staggered_multi_step' for alternating stepping scheme, 'staggered_one_step' is a one step scheme |

It generates the following output variables called by FractureProblem.variables
| output | Description |
|-----------|-------------|
| sol | solution vector of displacement and damage values for each timestep |
| eigs | vector of eigenvalue lambda1 at each timestep |
| eigs_der | ### deprecated, doesnt return anything now 
| der | design derivatives of sol for each timestep | 
| umax | history variable vector of displacement for each timestep |
| umaxder | design derivarive of history variable vector of displacement for each timestep |

---
## 2. gd_ex_spring_hpc_new.py -*hpc*-
Performs optimization runs from random starting points in the domain "[0.2,2]^2" on an objective with a spring problem as underlying state problem to be solved. 
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
| fmin | Function that returns for each input the value of f, if called with only one argument; values (f, df) if called with two inputs; values (f,df,G) for each input |
| gradient_function | ### deprecated;  set to NONE
| iterations | number of iterations |
| epsilon | List of widths of belt around discontinuity for smoothing for each level set function; if only one entry, then assumed to be same for all |
| kappa | List of widths of interior belt with max smoothing line length for each level set function; if only one entry, then assumed to be same for all |
| aleph | Initial search length for Armijo |

## NOTE: epsilon AND kappa ARE SQUARED INTERNALLY, SO HAND OVER sqrt(epsilon) and sqrt(kappa)

### Output

| Variable | Notes |
|----------|------|
| sol | Array of iterates for a given run |
| x | Count of function evaluations per iteration for a given run |

---

## 3. load_runs_ex_spring_hpc_new.py -*hpc*-

- Computes function values at iterates for ASM and GSM.
- processes output data from "gd_ex_spring_hpc_new.py" and stores the following output in "Run_XVals_hybrid.npy" and "Run_XVals_stoch.npy" for X-th run of ASM and GSM, respectively

| Variable | Notes | 
|---------|-------|
| Fsol_store | Array of values of either ASM or GSM at each iterate for run X | 
| x_store | Array of x function evaluations count for each iterate for run X |

---

## 4. plot_runs_ex_spring_hpc_new.py 

- Calculates mean, standard deviation, and prints values of "c_mean", "k_mean", and "c_mean^step".
- Plots the mean and standard deviation over optimization iterations
- **do_work**: Set to True to assemble output data from "load_runs_ex_spring_hpc_new.py" and store the following output in "Vals_hybrid.npy" and "Vals_stoch.npy" for ASM and GSM, respectively

| Variable | Notes | 
|---------|-------|
| Fsol_store | Array of assembled values of either ASM or GSM at each iterate for each run | 
| x_store | Array of assembled function evaluations count for each iterate for each run |

---

## 5. load_landscape_ex_spring_hpc.py -*hpc*-

Generates landscapes for ASM, GSM, and the discontinuous objective function. 
For process umber X stores the following in "codeX_Regular_Landscape.npy", "codeX_Glob_Landscape.npy" and "codeX_Adap_Landscape.npy" for f, GSM and ASM,respectively.
**code** is an internally set parameter that assigns an id to the filename

codeX_Glob_Landscape.npy --- stores landscape data generated for GSM for process X
| Variable | Notes | 
|---------|-------|
| x_1 | matrix of x[1] meshgrid values 
| x_2 | matrix of x[2] meshgrid values 
| vals_stoch | = list of function evaluations of GSM at each (x[1],x[2]) coordinate assigned to the X-th process.

codeX_Adap_Landscape.npy --- stores landscape data generated for ASM for process X
| Variable | Notes | 
|---------|-------|
| x_1 | matrix of x[1] meshgrid values 
| x_2 | matrix of x[2] meshgrid values 
| vals_adap | list of function evaluations of ASM at each (x[1],x[2]) coordinate assigned to the X-th process.
| grads |  list of smoothing direction (when smoothing is performed) at given (x[1],x[2]) coordinate assigned to the X-th process.
| eps_val |  chosen epsilon value for ASM 
| kappa |  chosen kappa value for ASM

codeX_Regular_Landscape.npy --- stores landscape data generated for f and G for process X
| Variable | Notes | 
|---------|-------|
| x_1 | matrix of x[1] meshgrid values 
| x_2 | matrix of x[2] meshgrid values 
| vals | list of function evaluations of f at each (x[1],x[2]) coordinate assigned to the X-th process.
| eigs | list of function evaluations of G(in this case lambda_1) at each (x[1],x[2]) coordinate assigned to the X-th process.

---

## 6. plot_ls_ex_spring_new.py
- Plots the landscapes for ASM, GSM, and the discontinuous objective function from assembled landscape files
- **do_work**: Set to True to assemble output data from "load_landscape_ex_spring_hpc.py" and store the following output in "code_Smooth_Landscape.npy" for f, GSM and ASM.

| Variable | Notes | 
|---------|-------|
| vals | list of function evaluations of f at each (x[1],x[2]) coordinate
| vals_g |  list of function evaluations of GSM at each (x[1],x[2]) coordinate
| vals_ad |  list of function evaluations of ASM at each (x[1],x[2]) coordinate
| x_1 |  matrix of x[1] meshgrid values 
| x_2 |  matrix of x[2] meshgrid values 
| E |  list of function evaluations of G(in this case lambda_1) at each (x[1],x[2]) coordinate

---

## How to Run simulations: 

1. Adjust "Spring_fct" in "gd_ex_spring_hpc_new.py" to set your objective.
2. Set parameters in "gd_ex_spring_hpc_new.py", including metadata and state problem parameters.
3. Run "gd_ex_spring_hpc_new.py".
4. Use "load_runs_ex_spring_hpc_new.py" to compute values for each run.
5. Use "plot_runs_ex_spring_hpc_new.py" to assemble values and generate plots for specified dataset.

6a. Use "load_landscape_ex_nGs.py" to generate landscape files for specified dataset, set parameters in file.
6b. Use "plot_ls_ex_spring_new.py" to assemble and visualize landscape files for specified dataset.


The default folder in "gd_ex_spring_hpc_new.py" and any methods that generates data is set to 'test', so nothing is overwritten by accident. 
The default value of "do_work=False" in any file using that Boolean for the same reason.
