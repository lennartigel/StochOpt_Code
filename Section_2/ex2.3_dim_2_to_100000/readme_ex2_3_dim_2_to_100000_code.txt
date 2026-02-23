# ASM vs GSM Optimization Experiments

This repository contains simulation code to compare the **Adaptive Smoothing Method (ASM)** and **Global Smoothing Methods (GSM)**. 

---

## 1. gd_ex2_100d.py

Performs optimization runs from random starting points in the domain "[-20,20]^N". 
You can adjust the minima of the objective by changing "xmin1".

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

### The following parameters are internally set
| Parameter | Description |
|-----------|-------------|
| tol | Small offset to avoid instability of gradient on smoothing set boundary |
| epsilon | Width of belt around discontinuity for smoothing |
| kappa | Width of interior belt with max smoothing line length |
~~~ Armijo Parameters
| Parameter | Description |
|-----------|-------------|
| theta | Fixed at 0.001 |
| aleph | Initial search length |
| omega | Shrinking factor fixed at 0.5 |

###Optimization is performed by calling the optimization function of GradientDescent with the following inputs.
| Input | Description |
|-----------|-------------|
fmin | Function that returns the value of f for each input 
gradient_function | Function that returns the value of nablaf for each input, set to NONE for GSM
iterations | number of iterations

---

### Output

| Variable | Notes |
|----------|------|
| sol | Array of iterates for a given run |
| x | Count of function evaluations per iteration for a given run |

---

## 2. load_runs_ex2_100d.py

- Computes function values at iterates for ASM and GSM.
- Calculates mean, standard deviation, and prints values of "c_mean", "k_mean", and "c_mean^step".
- Computes the MLE of function evaluations for increasing dimensions for the given datasets. 
- **do_work**: Set to True to process output data from "gd_ex2_100d.py" and store the following output in "Vals_hybrid.npy" for ASM, respectively

| Variable | Notes | 
|---------|-------|
| Fsol_store | Array of values of either ASM or GSM at each iterate for each run | 
| x_store | Array of x function evaluations count for each iterate for each run |

---

## 5. How to Run simulations: 

1. Adjust "xmin1" for your objective.
2. Set general parameters in "gd_ex2_100d.py".
3. Run "gd_ex2_100d.py".
4. Use "load_runs_ex2_100d.py" to compute statistics and generate plots for specified dataset.

The default folder in "gd_ex2_100d.py" is set to 'test', so nothing is overwritten by accident. 
The default value "do_work=False" in "load_runs_ex2_100d.py", so nothing is accidentally overwritten. Change it to generate data for specific data sets and landscapes.
