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
from matplotlib.colors import ListedColormap
from scipy.integrate import quad, dblquad
from scipy.special import exp1, gamma, k0, k1

### class for gradient descent scheme
class GradientDescent:
    def __init__(self, xstart,  sigma=1.0, npop=10, gradient_available = False, check_for_stability = False, use_one_directional_smoothing= False,npop_der = 0, npop_stoch = 0):
        self.xstart = np.asarray(xstart)
        self.mu = self.xstart.copy()
        self.npop = npop
        self.sigma = sigma
        self.sigma_step = 1.0
        self.max_step = sigma
        self.ndim = np.shape(xstart)[0]
        self.step_old = np.zeros(self.ndim)
        self.history = {"solution": [], 'NumIters':[], 'FuncEvals':[]}

        print('dimension number:',self.ndim)

        self.npop_der = npop_der
        self.npop_stoch = npop_stoch

        self.gradient_available = gradient_available
        self.check_for_stability = check_for_stability
        self.use_one_directional_smoothing = use_one_directional_smoothing
        self.iterates_even = [np.asarray((1,0)),np.asarray((1,1))]
        self.iterates_odd = [np.asarray((1,-1)),np.asarray((1,1))]

        if self.use_one_directional_smoothing:
            ### straight 1d 
            YY = self.recursive_function_odd(int(1/2)+1) ### only even dimensions allowed
            C = ( ( YY[0] * k1(1/2) - YY[1] * k0(1/2) )/(2 * np.exp(1/2)) ) * (2 * gamma(1+1/2) )**1 * 1/(gamma(1+1/2))
            XX = self.recursive_function_even(int(2/2)) ### only even dimensions allowed
            C2 = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**2 * 1/(gamma(1+2/2))
            print('C1',C, 'C2', C2)
            self.kernel = lambda x : 1/C * np.exp(-1/(1-x**2)) 
            self.kernel_der = lambda x : 1/C * np.exp(-1/(1-x**2))* 2 * x/(-1 + x**2)**2

        
        else:
            if self.ndim%2 ==0:
                XX = self.recursive_function_even(int(self.ndim/2)) ### only even dimensions allowed
                C = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            else:
                YY = self.recursive_function_odd(int(self.ndim/2) +1) ### only even dimensions allowed
                C = ( ( YY[0] * k1(1/2) - YY[1] * k0(1/2) )/(2 * np.exp(1/2)) ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            self.kernel = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2) )) if np.sum(x_n**2) <= 1 else 0
            self.kernel_der = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2))) * 2*x_n/(-1 + np.sum(x_n**2))**2 if np.sum(x_n**2) <= 1 else 0
            print(C)
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
            pointsNd = np.zeros(( int(self.npop),self.ndim))
            base_nd = np.linspace(-1,1,self.npop_stoch+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
            coords = tuple([base_nd]*self.ndim)
            mesh_nd = np.meshgrid(*coords, indexing = 'ij')
            for i in range(self.ndim):
                pointsNd[:,i] = np.reshape(mesh_nd[i], int(self.npop)) 
            self.pointsNd = pointsNd[np.where(np.sum(np.abs(pointsNd)**2, axis = 1) <=1)]
            
            
            self.num_samples = self.pointsNd.size
            
        return

    def generate_weights(self):
        if self.use_one_directional_smoothing:
            self.weights1d = np.asarray([self.kernel(x) for x in self.points1d])
            self.weights_der1d = np.asarray([self.kernel_der(x) for x in self.points1d])
            print('1d integral approximation:',2 * 1/(self.npop_stoch+1) * np.sum(self.weights1d))
            print('1d der integral approximation:', 2 * 1/(self.npop_stoch+1) * np.sum(self.weights_der1d))
        elif self.npop > 4:
            

            self.weightsNd = np.asarray([self.kernel(x) for x in self.pointsNd])
            self.weights_derNd = np.asarray([self.kernel_der(x) for x in self.pointsNd])

            print('Nd integral approximation:',2**self.ndim * 1/(int(self.npop_stoch)+1)**self.ndim * np.sum(self.weightsNd))
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
    def ask_1d(self, center, length):
        if self.gradient_available:
            y =np.ones((self.npop,self.ndim))*center
            y = list(y)
            y.append(center)
        else:
            d = length * self.points1d + center[0]
            y = np.ones((self.npop,self.ndim))
                        
            ### sample along line
            y[:,1:] = center[1:]
            y[:,0] = d
            y = list(y)
        return y

    def compute_val(self, inp):
        ### computes the function value for smoothed functions
        
        if self.gradient_available:
            val = self.obj([inp,inp])[-1]
        elif self.use_one_directional_smoothing:
            if inp[0]**2 >= self.eps_val- self.tol*self.eps_val:
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
            val = 2**self.ndim * 1/(int(self.npop_stoch)+1)**self.ndim * np.inner(f_y, self.weightsNd)
            

        return val

    def compute_grad(self, inp, function_values = None, gradient_values = None):
        ### computes the gradient value for smoothed functions
        if self.gradient_available:
            grad = gradient_values[0]
        elif self.use_one_directional_smoothing:
            if inp[0]**2 >= self.eps_val- self.tol*self.eps_val:
                gradient_values = self.obj_grad([inp,inp])
                return gradient_values[0]
            z = ((inp[0])**2-self.kappa)/(self.eps_val - self.kappa)
            zder =  ((inp[0])*2 * 1)/(self.eps_val - self.kappa)
            if z >= 0: 
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
            grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), (self.points1d[1:] * self.weights1d[1:]) )
            grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der1d)
            grad_0 = ( sigma_loc_der/sigma_loc * grad_0_front + 1/sigma_loc * grad_0_back)
            grad = np.zeros((self.ndim,))
            grad[0] = grad_0
            grad[1:] = gradient_values[1:]
        else:
            if function_values is None:
                y = self.ask(inp)
                function_values = self.obj(y)
            w = len(function_values)
            grad = 2**self.ndim * 1/(int(self.npop_stoch)+1)**self.ndim * 1/self.sigma * np.inner(function_values, self.weights_derNd.transpose())
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

        if self.Armijo and not unchanged_mu:
            ff_mu = self.compute_val(self.mu)
            m = np.inner(grad, step)
            self.eval_count +=1
            Armijo_iter = 1
            while ( self.compute_val(mu_aleph) > ff_mu + 0.001 * aleph * m ):
                aleph *= 0.5
                mu_aleph = self.mu.copy() + aleph * step
                Armijo_iter +=1
                self.eval_count += 1
                
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
        self.aleph = 5

        self.tol = 10**-3
        
        self.obj = lambda x: objective_function(x)
        if gradient_function is not None: 
            self.obj_grad = lambda x: gradient_function(x)

        self.generate_samples()
        self.generate_weights()
        while not self.stop() and iteration < iterations:
            ### if stability check, check whether close to discontinuity (only in 1d smoothing used)
            if self.check_for_stability:
                if np.where((self.mu[0])**2 < self.eps_val - self.tol*self.eps_val)[0].size > 0: ### check func val if close to x = 0
                    unstable = True ### set unstable flag true, indicated smoothing needed
                    self.npop = self.npop_stoch ### set number of sampling to smoothing number
                    self.num_samples = self.npop_stoch
                    self.gradient_available = False ### indicate you need approximate gradient from method
                    if self.use_one_directional_smoothing:
                        ### smoothing only in one direction
                        z = ((self.mu[0])**2-self.kappa)/(self.eps_val - self.kappa)
                        if z >= 0: 
                            sigma_loc = self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
                        else:
                            sigma_loc = self.sigma
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
            #print(f"Iteration: {iteration}, Solution. {self.mu}") ### prints current func val if you want 
            self.history["solution"] += [self.mu.copy()]
            iteration += 1
        return self


if __name__ == "__main__":

    for ndim in [2,10,50,10**2,5*10**2, 10**3,5*10**3,10**4, 5*10**4, 10**5]:
    
        xmin1 = np.zeros((ndim,))

        def abs_sphere(x, args=()):
            ### discontinuous absolute value function, with instant drop at x= 0
            x = np.reshape(x, (-1, x[0].shape[-1]))
            f_x = np.where(
                x[:, 0] > 0.0,
                1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
                -1 + 0.1*np.absolute(x[:,0] - xmin1[0]) + 0.01 * np.sum((x[:,1:] - xmin1[1:])**2, axis = 1 ),
            )
            return f_x
        
        def abs_gradient(x, args=()):
            ### gradient function for discontinuous absolute value function
            x = np.reshape(x, (-1, x[0].shape[-1]))
            df_x = np.zeros(x.shape)        
            df_x[np.where(x[:, 0] > 0.0),0] = 0.1 
            df_x[np.where(x[:, 0] <= 0.0),0] = -0.1 
            df_x[:,1:] = 0.02 * (x-xmin1)[:,1:]
            return df_x

        ### define objective functions
        fmin = abs_sphere
        grad_min = abs_gradient

        ### meta data for optimization
        iter_max = 100 ### max number of optim steps
        multi = 1 ### multiplication factor to balance number of func evals between methods
        max_try = 100 ### max number of random runs

        
        pop_per_dir = 15
        sigma = 1*1.0 ### size of standard integral
        
        ## folder name
        pathing = './comp_big_new/Ex2_100d_' + str(ndim)
        if not os.path.exists(pathing):
            os.makedirs(pathing)

        ### meta data file
        metadata = pathing + '/metadata.npy'
        with open(metadata, 'wb') as f:
                np.savez(f,iter_max = iter_max, multi= multi, max_try = max_try, sigma = sigma)
        
        for i in range(max_try):
            print('------------------- Iteration %i -------------------'%i)
            filename3 = pathing + '/Run_' + str(i) + '_results3.npy'

            ### starting point (random or deterministic, your choice)
            
            ### for random runs
            xstart = 20*( np.random.rand(ndim)*2-1)
            
            print('start point:',xstart)

            
            print('##########################Optimizating Hybrid Step %i dim: %i ##########################'%(i,ndim))
            opt3 = GradientDescent(
                xstart=xstart, npop=1, sigma=sigma,  gradient_available = True, 
                check_for_stability = True, use_one_directional_smoothing = True, npop_der = 1,npop_stoch = pop_per_dir
            )
            opt3.optimize(fmin, iterations=iter_max*multi, gradient_function = grad_min)
            sol3 = opt3.history["solution"]
            sol3 = np.asarray(sol3)
            x3 = opt3.history['NumIters']
            
            with open(filename3, 'wb') as f:
                np.savez(f, sol3 = sol3, x3 = x3)    
            print('end point:',sol3[-1])
