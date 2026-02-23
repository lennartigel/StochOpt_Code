############################################
####script to optimize a discontinuous objective functional built from two absolute value functions by
####three different methods, full smoothing (i.e stochastic optimization),
####normal gradient descent with point evaluation,
####and a hybrid method where smoothing is only performed near the discontinuity.
####The discontinuity is detected via checking whether the function value has dropped under a threshhold
####as we have set up the objective to have the discontinuity at the axis x = 0
############################################

import numpy as np
import os
import datetime
from scipy.integrate import quad, dblquad
from scipy.special import exp1, gamma, k0, k1


### class for gradient descent scheme
class GradientDescent:
    def __init__(self, xstart,  sigma=1.0, npop=10, gradient_available = False, check_for_stability = False, use_one_directional_smoothing= False,npop_der = 0, npop_stoch = 0):
        self.xstart = np.asarray(xstart)
        self.mu = self.xstart.copy()
        self.npop = npop
        self.dim = len(xstart)
        self.sigma = sigma
        self.sigma_step = 1.0
        self.max_step = sigma
        self.step_old = np.zeros(self.dim)
        self.history = {"solution": [], 'NumIters':[], 'FuncEvals':[]}

        self.npop_der = npop_der
        self.npop_stoch = npop_stoch

        self.gradient_available = gradient_available
        self.check_for_stability = check_for_stability
        self.use_one_directional_smoothing = use_one_directional_smoothing

        self.softmax = lambda a,b: (a+b+np.sqrt((a-b)**2 +0.001))/2

        self.iterates_even = [np.asarray((1,0)),np.asarray((1,1))]
        self.iterates_odd = [np.asarray((1,-1)),np.asarray((1,1))]
        
        XX = self.recursive_function_even(int(1)) ### only even dimensions allowed
        C = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**2 * 1/(gamma(1+2/2))
        
        print(C)
        self.kernel = lambda x,y: 1/C * np.exp(-1/(1-x**2 - y**2))
        self.kernel_dx = lambda x,y: 1/C * np.exp(-1/(1-x**2 - y**2)) * (2*x)/(-1+x**2+y**2)**2
        self.kernel_dy = lambda x,y: 1/C * np.exp(-1/(1-x**2 - y**2)) * (2*y)/(-1+x**2+y**2)**2

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
            self.points1d = np.linspace(-1,1,self.npop_stoch+2)[1:-1]
            self.num_samples = self.points1d.size
        elif self.npop > 4:
            points2d = np.zeros(( int(self.npop),2))
            base_2d = np.linspace(-1,1,int(np.sqrt(self.npop))+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
            points2d[:,0] = np.kron(np.ones(int(np.sqrt(self.npop))), base_2d)
            points2d[:,1] = np.kron(base_2d,np.ones(int(np.sqrt(self.npop))))
            self.points2d = points2d[np.where(np.abs(points2d[:,0])**2 + np.abs(points2d[:,1])**2 < 1)]
            self.num_samples = self.points2d.size
        return

    def generate_weights(self):
        if self.use_one_directional_smoothing:
            quadf = lambda y: quad(lambda x: self.kernel(x,y), -np.sqrt(1-abs(y)**2), np.sqrt(1-abs(y)**2) )
            quaddf = lambda y: quad(lambda x: self.kernel_dy(x,y), -np.sqrt(1-abs(y)**2), np.sqrt(1-abs(y)**2) ) 
            self.weights1d = np.asarray([quadf(x)[0] for x in self.points1d])
            self.weights_der1d = np.asarray([quaddf(x)[0] for x in self.points1d])
            print('1d integral approximation:',2 * 1/(self.npop_stoch+1) * np.sum(self.weights1d))
        elif self.npop > 4:
            
            self.weights2d = self.kernel(self.points2d[:,0],self.points2d[:,1])
            self.weights_der2d_x =  self.kernel_dx(self.points2d[:,0],self.points2d[:,1])
            self.weights_der2d_y =  self.kernel_dy(self.points2d[:,0],self.points2d[:,1])

            print('2d integral approximation:',4 * 1/(int(np.sqrt(self.npop))+1)**2 * np.sum(self.weights2d))            
        return
        
    ### returns samples in 2d region
    def ask(self, center):
        ### if gradient exists only return point evaluation
        if self.gradient_available:
            y =np.ones((self.npop,2))*center
            y = list(y)
            y.append(center)
        ### if no gradient available return sampling points in stoch region
        else:
            y = self.sigma * self.points2d + center
            y = list(y)
        return y

    ### returns samples along 1d line
    def ask_1d(self, center, length):
        if self.gradient_available:
            y =np.ones((self.npop,2))*center
            y = list(y)
            y.append(center)
        else:
            d = length * self.points1d + center[0]
            y = np.ones((self.npop,2))
                        
            ### sample along line
            y[:,1] = center[1]
            y[:,0] = d
            y = list(y)
        return y

    def compute_val(self, inp):
        ### computes the function value for smoothed functions
        
        if self.gradient_available:
            val = self.obj([inp,inp])[-1]
        elif self.use_one_directional_smoothing:
            if inp[0]**2 >= self.eps_val-10**-3:
                val = self.obj([inp,inp])[-1]
                return val
            z = ((inp[0])**2-self.kappa)/(self.eps_val - self.kappa)
            if z >= 0: 
                sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
            else:
                sigma_loc = self.sigma
            y = self.ask_1d(inp, sigma_loc)
            f_y = self.obj(y)
            val = 2 * 1/(self.npop_stoch+1) * np.inner(f_y, self.weights1d)
        else:
            y = self.ask(inp)
            f_y = self.obj(y)
            val = 4 * 1/(int(np.sqrt(self.npop))+1)**2 * np.inner(f_y, self.weights2d) 

        return val

    def compute_grad(self, inp, function_values = None, gradient_values = None):
        ### computes the gradient value for smoothed functions
        if self.gradient_available:
            if gradient_values is None:
                gradient_values = self.obj_grad([inp,inp])
            grad = gradient_values[0]
        elif self.use_one_directional_smoothing:
            if inp[0]**2 >= self.eps_val-10**-3:
                gradient_values = self.obj_grad([inp,inp])
                return gradient_values[0]
            z = ((inp[0])**2-self.kappa)/(self.eps_val - self.kappa)
            zder =  ((inp[0])*2)/(self.eps_val - self.kappa)
            if z>= 0:
                sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                sigma_loc_der = - self.sigma * zder * ( 30*(1-z)**4 - 60*(1-z)**3 + 30*(1-z)**2)
            else:
                sigma_loc = self.sigma
                sigma_loc_der = 0
            if function_values is None:
                y = self.ask_1d(inp, sigma_loc)
                function_values = self.obj(y)
                gradient_values = self.obj_grad([inp,inp])[-1]

            w = len(function_values)

            grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), self.points1d[1:] * self.weights1d[1:])
            grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der1d)
            grad_0 = ( sigma_loc_der/sigma_loc * grad_0_front + 1/sigma_loc * grad_0_back)
            grad_1 = gradient_values[1]
            grad = np.array([grad_0, grad_1])
        else:
            if function_values is None:
                y = self.ask(inp)
                function_values = self.obj(y)
            w = len(function_values)
            grad_0 = 4 * 1/(int(np.sqrt(self.npop))+1)**2 * 1/self.sigma * np.inner(function_values, self.weights_der2d_x)
            grad_1 = 4 * 1/(int(np.sqrt(self.npop))+1)**2 * 1/self.sigma * np.inner(function_values, self.weights_der2d_y)
            grad = np.array([grad_0, grad_1])
        return grad
        

    def tell(self, solutions, function_values, gradient_values = None):
        ### solutions contains sampling points (2d, 1d, or point eval)
        ### function_values contains func vals at sampling points (2d, 1d, or point eval)
        ### gradient_values are necessary in 1d smoothing, give exact grad at current point
        ###     and let you decompose into reliable and unreliable part, then build a combi version

        self.eval_count = 0

        grad = self.compute_grad(self.mu,function_values, gradient_values)
        step = - grad ### step size
        aleph = self.aleph
        mu_aleph = self.mu.copy() + aleph * step 
        m = np.inner(grad, step)
        if np.linalg.norm(grad) <= 10**-12:
            unchanged_mu = True
        else:
            unchanged_mu = False
            self.eval_count +=1

        if self.Armijo and not unchanged_mu:
            ff_mu = self.compute_val(self.mu)
            m = np.inner(grad, step)
            Armijo_iter = 1
            while ( self.compute_val(mu_aleph) > ff_mu + 0.001 * aleph * m ):
                aleph *= 0.5
                mu_aleph = self.mu.copy() + aleph * step
                Armijo_iter +=1
                self.eval_count +=1
                
        if self.Armijo and unchanged_mu:
            aleph = 0
            mu_aleph = self.mu.copy() 


        self.mu = mu_aleph.copy()
        self.step_old = step.copy()

    def stop(self):
        return False

    def result(self):
        return self.mu

    def optimize(self, objective_function, iterations=1000, gradient_function = None, args=()):
        iteration = 0
        
        self.eps_val = 4*10**-0
        self.kappa = self.eps_val/2
        self.Armijo = True
        self.aleph = 10
        self.tol = 10**-3
        
        self.obj = lambda x: objective_function(x)
        if gradient_function is not None: 
            self.obj_grad = lambda x: gradient_function(x)

        self.generate_samples()
        self.generate_weights()
        while not self.stop() and iteration < iterations:
            ### if stability check, check whether close to discontinuity (only in 1d smoothing used)
            if self.check_for_stability:
                if self.mu[0]**2 < self.eps_val - self.tol * self.eps_val:
                    unstable = True ### set unstable flag true, indicated smoothing needed
                    self.npop = self.npop_stoch ### set number of sampling to smoothing number
                    self.num_samples = self.npop_stoch
                    self.gradient_available = False ### indicate you need approximate gradient from method
                    if self.use_one_directional_smoothing:
                        ### smoothing only in one direction
                        z = ((self.mu[0])**2-self.kappa)/(self.eps_val - self.kappa)
                        if z>= 0:
                            sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                        else:
                            sigma_loc = self.sigma
                        #sigma_loc = 1+self.softmax(-1,-self.softmax(0,((self.mu[0])**2-self.kappa)/(self.eps_val - self.kappa)))
                        X = self.ask_1d(self.mu,sigma_loc)
                        f = objective_function(X)
                    else:
                        ### smoothing in any direction
                        X = self.ask(self.mu)
                        f = objective_function(X)
                else:
                    ### if not unstable no smoothing 
                    unstable = False
                    self.gradient_available = True
                    self.npop = self.npop_der
                    self.num_samples = self.npop_der
                    X = self.ask(self.mu)
                    f = objective_function(X)
            else: 
                X = self.ask(self.mu)
                f = objective_function(X)
                
            ### for general stoch methods, with no stab check
            if self.gradient_available:
                df = gradient_function(X)
                self.tell(X, f, df)
            elif self.use_one_directional_smoothing:
                df_1 = gradient_function(X)[0]
                self.tell(X, f, df_1)
            else:
                self.tell(X,f)

            ### save data of the optimization run (num iterations, num fun evals)
            if len(self.history['NumIters']) == 0:
                self.history['NumIters'].append(self.npop)
                self.history['FuncEvals'].append(self.npop)
            else: 
                self.history['NumIters'].append(self.history['NumIters'][-1] + self.eval_count*self.num_samples)
                self.history['FuncEvals'].append(self.npop)
            
            if self.check_for_stability and unstable:
                self.npop = self.npop_der
                unstable = False
                self.gradient_available = True
            self.history["solution"] += [self.mu.copy()]
            iteration += 1
        return self


if __name__ == "__main__":
    

    xmin1 = [0,0]
    xmin2 = [-3,3]

    def abs_sphere(x, args=()):
        ### discontinuous absolute value function, with instant drop at x= 0
        x = np.reshape(x, (-1, x[0].shape[-1]))
        f_x = np.where(
            x[:, 0] > 0.0,
            1 + 0.1*np.absolute(x - xmin1)[:,0] + 0.01 * (x - xmin1)[:,1]**2,
            -1 + 0.1*np.absolute(x - xmin1)[:,0] + 0.01 * (x - xmin1)[:,1]**2,
        )
        return f_x
    
    def abs_gradient(x, args=()):
        ### gradient function for discontinuous absolute value function
        x = np.reshape(x, (-1, x[0].shape[-1]))
        df_x = np.zeros(x.shape)        
        df_x[np.where(x[:, 0] > 0.0),0] = 0.1 
        df_x[np.where(x[:, 0] <= 0.0),0] = -0.1 
        df_x[:,1] = 0.02 * (x-xmin1)[:,1]
        return df_x

    ### define objective functions
    fmin = abs_sphere
    grad_min = abs_gradient

    ### meta data for optimization
    iter_max = 100 ### max number of optim steps
    multi = 10 ### multiplication factor to balance number of func evals between methods
    max_try = 100 ### max number of random runs

    ### meta data for single run optimization
    iter_max = 100
    multi = 10
    max_try = 1

    
    pop_per_dir = 10
    sigma = 1*1.0 ### size of standard integral
    
    ### storage matrices/vectors
    Fsol_store = np.zeros((iter_max,max_try))
    Fsol3_store = np.zeros((iter_max*multi,max_try))
    x_store = np.zeros(((iter_max,max_try)))
    x3_store = np.zeros(((iter_max*multi,max_try)))
    
    ## folder name
    date_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    date_time = 'final_new'
    date_time = 'final_ex_new'  ### for example run
    pathing = './Ex2_' + date_time
    if not os.path.exists(pathing): 
        os.makedirs(pathing)

    ### meta data file
    metadata = pathing + '/metadata.npy'
    with open(metadata, 'wb') as f:
            np.savez(f,iter_max = iter_max, multi= multi, max_try = max_try, sigma = sigma)
    
    for i in range(max_try):
        ## folder name
        print('------------------- Iteration %i -------------------'%i)
        pathing = './Ex2_' + date_time
        filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
        filename3 = pathing + '/Run_' + str(i) + '_results3.npy'

        ### starting point (random or deterministic, your choice)

         ### for deterministic runs
        xstart=[9,9]
        ### for random runs
##        xstart = 20*( np.random.rand(2)*2-1)
        
        print('starting point:', xstart)

        ### optimization methods
        opt = GradientDescent(
            xstart=xstart, npop=pop_per_dir*pop_per_dir, sigma=sigma, gradient_available = False
        )
        opt3 = GradientDescent(
            xstart=xstart, npop=1, sigma=sigma,  gradient_available = True, 
            check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,npop_stoch = pop_per_dir
        )
        
        ### start optimization
        print('##########################Optimizating Stochastic Step %i ##########################'%i)
        opt.optimize(fmin, iterations= iter_max, gradient_function = grad_min)
        print('##########################Optimizating Hybrid Step %i ##########################'%i)
        opt3.optimize(fmin, iterations=iter_max*multi, gradient_function = grad_min)
        sol = opt.history["solution"]
        sol = np.asarray(sol)
        sol3 = opt3.history["solution"]
        sol3 = np.asarray(sol3)
                
        ### read out results
        x = opt.history['NumIters']
        x3 = opt3.history['NumIters']

        ### save results to file
        with open(filename3, 'wb') as f:
            np.savez(f, sol3 = sol3, x3 = x3)
        with open(filename1, 'wb') as f:
            np.savez(f, sol1 = sol, x1 = x)
     
