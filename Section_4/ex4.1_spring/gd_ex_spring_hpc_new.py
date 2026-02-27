import numpy as np
import os
import datetime
from Springclass_hpc_AS import FractureProblem
from scipy.integrate import quad, dblquad
from scipy.special import exp1, gamma, k0, k1

class GradientDescent:
    def __init__(self, xstart, sigma=1.0, npop=10, gradient_available = False, check_for_stability = False, use_one_directional_smoothing= False,npop_der = 0, npop_stoch = 0, ident = None):
        self.xstart = np.asarray(xstart)
        self.mu = self.xstart.copy()
        self.npop = npop
        self.dim = len(xstart)
        self.sigma = sigma
        self.sigma_step = 1.0
        self.max_step = sigma
        self.step_old = np.zeros(self.dim)
        self.history = {"solution": [], 'NumIters':[], 'FuncEvals':[], 'smoothing_directions': []}

        self.gradient_available = gradient_available
        self.check_for_stability = check_for_stability
        self.use_one_directional_smoothing = use_one_directional_smoothing

        self.direction_grad = None

        self.npop_der = npop_der
        
        self.ident = ident
        
        #### specifically made for 2d, wont work for nd
        self.npop_stoch = npop_stoch
        
        self.ndim = np.shape(xstart)[0]
        
        self.eval_count = 0
        
        self.iterates_even = [np.asarray((1,0)),np.asarray((1,1))]
        self.iterates_odd = [np.asarray((1,-1)),np.asarray((1,1))]
        
        if self.use_one_directional_smoothing:
            XX = self.recursive_function_even(int(1)) ### only even dimensions allowed
            C = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**2 * 1/(gamma(1+2/2))
            base_rho = lambda x,y: 1/C * np.exp(-1/(1-x**2 - y**2))
            base_rho_der = lambda x,y: 1/C * np.exp(-1/(1-x**2 - y**2)) * (2*y)/(-1+x**2+y**2)**2
            self.kernel = lambda y: quad(lambda x: base_rho(x,y), -np.sqrt(1-abs(y)**2), np.sqrt(1-abs(y)**2) )[0]
            self.kernel_der = lambda y: quad(lambda x: base_rho_der(x,y), -np.sqrt(1-abs(y)**2), np.sqrt(1-abs(y)**2) )[0]
        
        else:
            if self.ndim%2 ==0:
                XX = self.recursive_function_even(int(self.ndim/2)) ### only even dimensions allowed
                C = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            else:
                YY = self.recursive_function_odd(int(self.ndim/2) +1) ### only even dimensions allowed
                C = ( ( YY[0] * k1(1/2) - YY[1] * k0(1/2) )/(2 * np.exp(1/2)) ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            self.kernel = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2) )) if np.sum(x_n**2) < 1 else 0
            self.kernel_der = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2))) * 2*x_n/(-1 + np.sum(x_n**2))**2 if np.sum(x_n**2) < 1 else 0
        return

    def recursive_function_even(self, ndim):
        if len(self.iterates_even) < ndim+1:
            new_iterate = (2*ndim-1)/(ndim-1) * self.recursive_function_even(ndim-1) - self.recursive_function_even(ndim-2)
            self.iterates_even.append(new_iterate)
            return new_iterate
        else:
            return self.iterates_even[ndim]
        
    def recursive_function_odd(self, ndim):
        if len(self.iterates_odd) < ndim+1:
            new_iterate = (4*ndim-4)/(2*ndim-3) * self.recursive_function_odd(ndim-1) - self.recursive_function_odd(ndim-2)
            self.iterates_odd.append(new_iterate)
            return new_iterate
        else:
            return self.iterates_odd[ndim]

    ### pre-generates samples unuformly distributed on [-sigma,sigma]
    def generate_samples(self):
        if self.use_one_directional_smoothing:
            
            self.points1d = np.linspace(-1,1,self.npop_stoch+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
        elif self.npop > 4:
     
            pointsNd = np.zeros(( int(self.npop),self.ndim))
            base_nd = np.linspace(-1,1,self.npop_stoch+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
            coords = tuple([base_nd]*self.ndim)
            mesh_nd = np.meshgrid(*coords, indexing = 'ij')
            
            for i in range(self.ndim):
                pointsNd[:,i] = np.reshape(mesh_nd[i], int(self.npop)) 
            self.pointsNd = pointsNd[np.where(np.sum(np.abs(pointsNd)**2, axis = 1) <1)]
            
            self.num_samples = self.pointsNd.size
        return

    def generate_weights(self):

        if self.use_one_directional_smoothing:

            self.weights1d = np.asarray([self.kernel(x) for x in self.points1d])
            self.weights_der1d = np.asarray([self.kernel_der(x) for x in self.points1d])
            
        elif self.npop > 4:

            self.weightsNd = np.asarray([self.kernel(x) for x in self.pointsNd])
            self.weights_derNd = np.asarray([self.kernel_der(x) for x in self.pointsNd])
            
        return

     ### returns samples in 2d region
    def ask(self, center):
        ### if gradient exists only return point evaluation
        if self.gradient_available:
            y =np.ones((self.npop,self.ndim))*center
            y = list(y)
            y.append(center)
        ### if no gradient available return sampling points in stoch region
        else:
            y = self.sigma * self.pointsNd + center
            y = list(y)
        return y


    ### returns samples along 1d line
    def ask_1d(self, center, length, *args):
        if self.gradient_available:
            y =np.ones((self.npop,self.ndim))*center
            y = list(y)
            y.append(center)
        else:
            direction = args[0]

            y = np.ones((self.npop_stoch,self.ndim))
            for i in range(self.ndim):
                y[:,i] = length * self.points1d * direction[i] + center[i]
                        
            ### sample along line
            y = list(y)
        return y
    

    def compute_val(self, inp, sigma_loc = None, func_vals = None, *args):
        ### computes the function value for smoothed functions
        
        if self.gradient_available:
            if func_vals is None:
                val = self.obj([inp])[-1]
            else:
                val = func_vals[-1]
        elif self.use_one_directional_smoothing:
            if sigma_loc is None:
                (f,df,E) = self.obj_eigs([inp]) ### get fun val, gradient val, min eigenvalue
                if E[-1]**2 < self.eps_val-self.tol*self.eps_val:
                    z = ((E[-1])**2-self.kappa)/(self.eps_val - self.kappa)
                    if z > 0: 
                        sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                    else: 
                        sigma_loc = self.sigma
                else: 
                    return f
                dE = self.compute_smoothing_direction(inp)
                direction  = dE[-1]/np.linalg.norm(dE[-1])
                y = self.ask_1d(inp, sigma_loc, direction)
                self.eval_count += self.npop_stoch
                f_y = self.obj(y)
                val = 2 * 1/(self.npop_stoch+1) * np.inner(f_y, self.weights1d)
            else: 
                direction = args[0]
                if func_vals is None:
                    y = self.ask_1d(inp, sigma_loc, direction)
                    f_y = self.obj(y)
                    self.eval_count += self.npop_stoch
                else:
                    f_y = func_vals
                val = 2 * 1/(self.npop_stoch+1) * np.inner(f_y, self.weights1d)
        else:
            if func_vals is None:
                y = self.ask(inp)
                f_y = self.obj(y)
                self.eval_count += self.npop
            val = 2**self.ndim * 1/(int(self.npop_stoch)+1)**self.ndim * np.inner(f_y, self.weightsNd)
        return val

    def compute_grad(self, inp, function_values = None, gradient_values = None, sigma_loc = None, sigma_loc_der = None, *args):
        ### computes the gradient value for smoothed functions
        if self.gradient_available:
            if function_values is None:
                (val,grad) = self.obj_grad([inp])
                grad = grad[0]
                print('grad#############################', grad)
            else:
                grad = gradient_values[0]
                
        elif self.use_one_directional_smoothing:
            if sigma_loc is None:
                (f,df,E) = self.obj_eigs([inp]) ### get fun val, gradient val, min eigenvalue
                if E[-1]**2 < self.eps_val-self.tol*self.eps_val:
                    z = ((E[-1])**2-self.kappa)/(self.eps_val - self.kappa)
                    if z>= 0:
                        sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                    else:
                        sigma_loc = self.sigma
                else: 
                    return df
                dE = self.compute_smoothing_direction(inp)
                zder =  ((E[-1])*2 * np.linalg.norm(dE[-1]))/(self.eps_val - self.kappa)
                if z>= 0:
                    sigma_loc_der = - self.sigma * zder * ( 30*(1-z)**4 - 60*(1-z)**3 + 30*(1-z)**2)
                else:
                    sigma_loc_der = 0
                ### get smoothing direction then compute perpendicular direction
                direction = dE[-1]/np.linalg.norm(dE[-1])
                perp_dir = np.array([direction[1], -direction[0]])/np.linalg.norm([direction[1], -direction[0]])
                ### smooth function vals along smoothing direction
                y = self.ask_1d(inp, sigma_loc,direction)
                function_values = self.obj(y)
                self.eval_count += self.npop_stoch
                ### obj func gradient at point
                (vg,gradient_values) = self.obj_grad([inp])[-1]
                gradient_values = gradient_values[0]
                ### assemble gradient              
                grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), self.points1d[1:] * self.weights1d[1:])
                grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der1d)

                grad_1 = ( sigma_loc_der/sigma_loc * grad_0_front + 1/sigma_loc * grad_0_back)

                grad_2 = np.dot(gradient_values, perp_dir)
                grad = (1/np.linalg.norm(direction)**2
                        * np.asarray([direction[0] * grad_1 + direction[1] * grad_2,
                                      direction[1] * grad_1 - direction[0] * grad_2])
                        )
            else:
                direction = args[0]
                perp_dir = np.array([direction[1], -direction[0]])/np.linalg.norm([direction[1], -direction[0]])

                grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), self.points1d[1:] * self.weights1d[1:])
                grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der1d)

                grad_1 = ( sigma_loc_der/sigma_loc * grad_0_front + 1/sigma_loc * grad_0_back)                
                grad_2 =  np.dot(gradient_values, perp_dir)
                grad = (1/np.linalg.norm(direction)**2
                        * np.asarray([direction[0] * grad_1 + direction[1] * grad_2, direction[1] * grad_1 - direction[0] * grad_2])
                        )

        else:
            if function_values is None:
                y = self.ask(inp)
                function_values = self.obj(y)
            w = len(function_values)
            
            grad = 2**self.ndim * 1/(int(self.npop_stoch)+1)**self.ndim * 1/self.sigma * np.inner(function_values, self.weights_derNd.transpose())

        return grad

    def compute_smoothing_direction(self, inp):
        dir_1 = np.zeros((2)) ## x direction vector
        dir_2 = np.zeros((2)) ## y direction vector
        dir_1[0] = 1
        dir_2[1] = 1
        grad_direction = np.zeros((2))
        self.eval_count += 4
        
        ## compute the direction of the discontinuity from the gradient of the smallest eigenvalue
        (ta,dta,E_fwd_1) = self.obj_eigs([inp+ dir_1*self.sten_size ])
        (ta,dta,E_bwd_1) = self.obj_eigs([inp - dir_1*self.sten_size ])
        (ta,dta,E_fwd_2) = self.obj_eigs([inp + dir_2*self.sten_size ])
        (ta,dta,E_bwd_2) = self.obj_eigs([inp - dir_2*self.sten_size ])
        
        ### construct gradient direction of min eigenvalue
        grad_direction[0] = (E_fwd_1[0] - E_bwd_1[0])/(2*self.sten_size)
        grad_direction[1] = (E_fwd_2[0] - E_bwd_2[0])/(2*self.sten_size)
        
##        return grad_direction
        return [grad_direction]

    def tell(self, solutions, function_values, gradient_values = None):
        ### solutions contains sampling points (2d, 1d, or point eval)
        ### function_values contains func vals at sampling points (2d, 1d, or point eval)
        ### gradient_values are necessary in 1d smoothing, give exact grad at current point
        ###     and let you decompose into reliable and unreliable part, then build a combi version
        
        self.eval_count += 0

        if not self.unchanged_mu:
            if self.use_one_directional_smoothing and self.sigma_loc_mu >0:
                grad = self.compute_grad(self.mu,function_values, gradient_values, self.sigma_loc_mu,self.sigma_loc_mu_der, self.grad_dir)
            elif self.use_one_directional_smoothing and self.sigma_loc_mu ==0: 
                grad = gradient_values
            else:
                grad = self.compute_grad(self.mu,function_values, gradient_values)
            if np.linalg.norm(grad) > self.aleph:
                grad = grad/np.linalg.norm(grad)
                step = - grad ### step direction
                aleph = self.aleph
                
            else:
                step = - grad ### step direction
                aleph = 1
            mu_aleph = self.mu.copy() + aleph * step
            m = np.inner(grad, step)

            if np.linalg.norm(grad) < 10**-6 and not self.unchanged_mu:
                self.unchanged_mu = True#
                print(self.ident,'gradient small:', np.linalg.norm(grad))

            ### armijo iteration
            if self.Armijo and not self.unchanged_mu:
                if self.use_one_directional_smoothing and self.sigma_loc_mu >0:
                    ff_mu = self.compute_val(self.mu, self.sigma_loc_mu, function_values, self.grad_dir)
                else:
                    ff_mu = self.compute_val(self.mu)
                m = np.inner(grad, step)
                Armijo_iter = 1
                ff_mu_try = self.compute_val(mu_aleph)
                while ( ff_mu_try > ff_mu + 10**-3 * aleph * m ):
                    aleph *= 0.5
                    mu_aleph = self.mu.copy() + aleph * step
                    Armijo_iter +=1
                    ff_mu_try = self.compute_val(mu_aleph)
                    if Armijo_iter >= 30: 
                        self.unchanged_mu = True
                        mu_aleph = self.mu.copy()
            self.mu = mu_aleph.copy()
            self.step_old = step.copy()

        else:
            aleph = 0
            mu_aleph = self.mu.copy() 

    def stop(self):
        return False

    def result(self):
        return self.mu

    ### optimization outer shell
    def optimize(self, objective_function, iterations=1000, gradient_function = None, args=(),Armijo = True,eps_val = 2, kappa = 1, aleph = 0.3):
        ### objective function should return for a given vector of sampling points
        ###     a vector of function values f
        ###     a vector of gradient values df (if 1d smoothing or point eval scheme)
        ###     a vector of the minimal eigenvalue E (over all time points) (if 1d smoothing)
        
        iteration = 0

        self.unchanged_mu = False
        self.tol = 10**-2

        self.eps_val = eps_val**2 
        self.kappa = kappa**2 
        self.Armijo = Armijo
        self.aleph = aleph
        
        self.obj = lambda x: objective_function(x)
        if self.gradient_available: 
            self.obj_grad = lambda x: objective_function(x, ['derivative'])
        if self.check_for_stability:
            self.obj_eigs = lambda x: objective_function(x, ['eigs', 'derivative'])

        self.sten_size = 10**-4 ### size of stencil  for approximation of eigenvalue derivative
        
        self.generate_samples()
        self.generate_weights()
        while iteration < iterations:
            print(self.ident,'iteration num: ',iteration, iterations)
            self.eval_count = 0
            if self.unchanged_mu:
                self.tell(self.mu,f)
            else:
                
                ### if stability check, check whether close to discontinuity (only in 1d smoothing used)
                if self.check_for_stability:
                    (fE,dfE,E) = self.obj_eigs([self.mu]) ### get fun val, gradient val, min eigenvalue
                    self.eval_count += self.npop_der
                    if E[-1]**2 < self.eps_val-self.tol*self.eps_val: ### check func val if close to x = 0
                        ## compute the direction of the discontinuity from the gradient of the smallest eigenvalue
                        dE = self.compute_smoothing_direction(self.mu)
                        self.grad_dir = dE[-1]/np.linalg.norm(dE[-1])
                        
                        ### choice between smoothing in 2d and 1d smoothing
                        ### performs smoothing in direction of eigval gradient
                        zder =  ((E[-1])*2 * np.linalg.norm(dE[-1]))/(self.eps_val - self.kappa)
                        z = ((E[-1])**2-self.kappa)/(self.eps_val - self.kappa)
                        if z >= 0:
                            self.sigma_loc_mu = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                            self.sigma_loc_mu_der = - self.sigma * zder * ( 30*(1-z)**4 - 60*(1-z)**3 + 30*(1-z)**2)
                        else:
                            self.sigma_loc_mu = self.sigma
                            self.sigma_loc_mu_der = zder * 0
                        X = self.ask_1d(self.mu,self.sigma_loc_mu,self.grad_dir)
                        self.history['smoothing_directions'].append(self.grad_dir)
                        f = objective_function(X)
                        self.eval_count += self.npop_stoch
                    else:
                        X = self.mu
                        self.sigma_loc_mu = 0
                        self.sigma_loc_mu_der =  0
                        f = fE[0]
                else:
                    ### for general stoch methods, with no stab check
                    X = self.ask(self.mu)

                
                ### generate gradient (or no gradient) depending method
                if self.gradient_available:
                    f = [fE,fE]
                    df = np.ones((2,2))*dfE
                    self.tell(X, f, df)
                elif self.use_one_directional_smoothing:
                    df_1 = np.asarray(dfE[0])
                    self.tell(X, f, df_1)
                else:
                    f = objective_function(X)
                    self.eval_count += self.npop
                    self.tell(X, f)

                
                    
            ### save data of the optimization run (num iterations, num fun evals)
            if len(self.history['NumIters']) == 0:
                self.history['NumIters'].append(self.eval_count)
                self.history['FuncEvals'].append(self.eval_count)
            else: 
                self.history['NumIters'].append(self.history['NumIters'][-1] + self.eval_count)
                self.history['FuncEvals'].append(self.npop)
            if not self.unchanged_mu:
                print(self.ident,'endresult: ',iteration, self.eval_count)
                
            self.history["solution"] += [self.mu.copy()]
            iteration += 1
        return self

def run_fun(run_number):
    

    n_time_steps = 101
    method = 'staggered_multi_step'
    Springprob = FractureProblem(n_time_steps, 1)
    

    def Spring_fct(x, args =()):
        ### calls spring simulation for function evaluation from script Springclass.py
        eigs = []
        vals = []
        derivs = []
        for entry in x:
            Springprob.G1 = entry[0]
            Springprob.G2 = entry[1]
            Springprob.solve(method = method)
            ### compute function value with solution from simulation
            vals.append(
                         - Springprob.sol[-2,-1] 
                        + (entry[0] - 0.7)**2
                        + (entry[1] - 0.9)**2
                        )
            ### compute derivative with sensitvities
            if len(args)>=1:
                
                derivs.append(
                              - Springprob.der[-2,:] 
                              +np.array([ 2*(entry[0] - 0.7), 0])
                              +np.array([ 0,2*(entry[1] - 0.9)])
                              )
            if len(args)>=2:
                ### eigenvalue 
                index = np.argmin(Springprob.eigs[0,:])
                eigs.append(Springprob.eigs[0,index]/(Springprob.G1 + Springprob.G2))

        
        if len(args)>=2:
            return np.asarray(vals), np.asarray(derivs),np.asarray(eigs)
        elif len(args)>=1:
            return np.asarray(vals), np.asarray(derivs)
        else: 
            return np.asarray(vals)
                 
    ### define objective functions
    fmin = Spring_fct
    fmincon = lambda x: Spring_fct(x,['Derivative'])

    ### meta data for optimization
    iter_max = 30 ### max number of optim steps
    multi = 10 ### multiplication factor to balance number of func evals between methods
    max_try = 32*4 ### max number of random runs
    
    sigma = 0.15*1.0 ### size of variance for normal distribution area for sampling
    eps_val = np.sqrt(0.7) ### theses values are squared inside GradientDescent class before optimization
    kappa = np.sqrt(0.5)

    aleph = 0.2 ### initial search length for armijo

    ### storage matrices/vectors
    Fsol3_store = np.zeros((iter_max*multi,max_try))
    x3_store = np.zeros(((iter_max*multi,max_try)))
    y3_store = np.zeros(((iter_max*multi,max_try)))

    Fsol2_store = np.zeros((iter_max,max_try))
    x2_store = np.zeros(((iter_max,max_try)))
    y2_store = np.zeros(((iter_max,max_try)))

    ### create folder structure for saving of data 
##    pathing = './ExAS_final_new'
    pathing = './test' ## set to test by default so nothing is overwritten by accident
    if not os.path.exists(pathing): 
            os.makedirs(pathing)
            
    ### meta data file
    metadata = pathing + '/metadata.npy'
    with open(metadata, 'wb') as f:
            np.savez(f,iter_max = iter_max, multi= multi, max_try = max_try, sigma = sigma)


    i = run_number
    ## folder name
    filename2 = pathing + '/Run_' + str(i) + '_results_stoch.npy'
    filename3 = pathing + '/Run_' + str(i) + '_results_hyb.npy'
    if not os.path.exists(pathing): 
        os.makedirs(pathing)

    ### starting point (random or deterministic, you choice)
    xstart = np.random.rand(2)*1.8 + 0.2
    print('------------------- Iteration %i -------------------'%i, xstart)

    npop = 100 ### for globally smoothing method
    npop_stoch = 10### for adaptively smoothing method

    ### optimization methods


    ################
    ### ASM
    opt3 = GradientDescent(
        xstart=xstart, npop=1, sigma=sigma, gradient_available = False, 
        check_for_stability = True, use_one_directional_smoothing = True,
        npop_der = 1,npop_stoch = npop_stoch, ident = run_number
    )
        
    ### start optimization 
    print('##########################Optimizating Hybrid Step %i ##########################'%i)
    opt3.optimize(fmin, iterations=iter_max*multi, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)
    ### read out results

    sol3 = opt3.history["solution"]
    sol3 = np.asarray(sol3)
    
    x3 = opt3.history['NumIters']
    nu3 = opt3.history['smoothing_directions']


    ### save results to file
    with open(filename3, 'wb') as f:
        np.savez(f,nu3 = nu3, sol3 = sol3, x3 = x3, eps_val = eps_val, kappa = kappa)

    ##################
    #### GSM
    opt2 = GradientDescent( xstart = xstart, sigma= sigma, npop=npop,
         gradient_available = False, check_for_stability = False,
         use_one_directional_smoothing= False,
         npop_der = 0, npop_stoch = npop_stoch, ident = run_number
    )

    ### start optimization 
    print('##########################Optimizating Stochastic Step %i ##########################'%i)
    opt2.optimize(fmin, iterations=iter_max, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)

    sol2 = opt2.history["solution"]
    sol2 = np.asarray(sol2)

    x2 = opt2.history['NumIters']
    
    with open(filename2, 'wb') as f:
        np.savez(f, sol2 = sol2, x2 = x2)
####
