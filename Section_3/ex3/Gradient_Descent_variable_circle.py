############################################
####two different methods, full smoothing (i.e stochastic optimization)
####and a hybrid method where smoothing is only performed near the discontinuity.
####The discontinuity is detected via checking the function of the minimal eigenvalue for the entire time evolution,
####    once it drops beneath a given threshhold, the direction of smoothing is detected dynamically via
####    the gradient of this function. 
############################################

import numpy as np
import os
import datetime
from scipy.integrate import quad, dblquad
from scipy.special import exp1, gamma, k0, k1
from scipy.linalg import null_space

### class for gradient descent scheme
class GradientDescent:
    def __init__(self, xstart, sigma=1.0, npop=10, gradient_available = False, check_for_stability = False, use_one_directional_smoothing= False,npop_der = 0, npop_stoch = 0, ident = None, num_id_funcs = 1):
        self.xstart = np.asarray(xstart)
        self.mu = self.xstart.copy()
        self.npop = npop
        self.dim = len(xstart)
        self.sigma = sigma
        self.sigma_step = 1.0
        self.max_step = sigma
        self.step_old = np.zeros(self.dim)
        self.history = {"solution": [], 'NumIters':[], 'FuncEvals':[], 'smoothing_directions': [], 'gradients': [], 'alphas': [], 'Armijo': []}

        self.gradient_available = gradient_available
        self.check_for_stability = check_for_stability
        self.use_one_directional_smoothing = use_one_directional_smoothing

        self.direction_grad = None

        self.npop_der = npop_der

        self.ident = ident

        self.stop_algorithm = False

        self.num_id_funcs = num_id_funcs
        print('number of level set functs:',self.num_id_funcs)

        
        #### specifically made for 2d, wont work for nd
        self.npop_stoch = npop_stoch
        
        self.ndim = np.shape(xstart)[0]
        print('dimension number:',self.ndim)
        
        self.eval_count = 0
        self.tol = 10**-3
        self.iterates_even = [np.asarray((1,0)),np.asarray((1,1))]
        self.iterates_odd = [np.asarray((1,-1)),np.asarray((1,1))]
        
        if self.use_one_directional_smoothing:
            ### choose between down integrated kernel from 2d or straight up 1d

            ### straight 1d 
            YY = self.recursive_function_odd(int(1/2)+1) ### only even dimensions allowed
            C = ( ( YY[0] * k1(1/2) - YY[1] * k0(1/2) )/(2 * np.exp(1/2)) ) * (2 * gamma(1+1/2) )**1 * 1/(gamma(1+1/2))
            XX = self.recursive_function_even(int(2/2)) ### only even dimensions allowed
            C2 = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**2 * 1/(gamma(1+2/2))
##            print('C1',C, 'C2', C2)
            kernel = lambda x : 1/C * np.exp(-1/(1-x**2)) 
            kernel_der = lambda x : 1/C * np.exp(-1/(1-x**2)) * 2 * x/(-1 + x**2)**2

            kernel2 = lambda x_n: 1/C2 * np.exp(-1/(1-np.sum(x_n**2) )) if np.sum(x_n**2) <= 1 else 0
            kernel2_der = lambda x_n: 1/C2 * np.exp(-1/(1-np.sum(x_n**2))) * 2*x_n/(-1 + np.sum(x_n**2))**2 if np.sum(x_n**2) <= 1 else 0

            self.kernel_list = [[]] * max(self.num_id_funcs,3)
            self.kernel_der_list = [[]] * max(self.num_id_funcs,3)

            self.kernel_list[0] = kernel
            self.kernel_der_list[0] = kernel_der

            self.kernel_list[1] = lambda xx: kernel2(xx)
            self.kernel_der_list[1] = lambda xx: kernel2_der(xx)
            self.kernel_list[2] = lambda xxx: kernel(xxx[0]) * kernel(xxx[1]) * kernel(xxx[2])
            self.kernel_der_list[2] = lambda xxx: np.array([kernel_der(xxx[0]), kernel_der(xxx[1]) , kernel_der(xxx[2]) ])
       

            
        else:
            if self.ndim%2 ==0:
                XX = self.recursive_function_even(int(self.ndim/2)) ### only even dimensions allowed
                C = ( 1/np.exp(1) * XX[0] - exp1(1) * XX[1] ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            else:
                YY = self.recursive_function_odd(int(self.ndim/2) +1) ### only odd dimensions allowed
                C = ( ( YY[0] * k1(1/2) - YY[1] * k0(1/2) )/(2 * np.exp(1/2)) ) * (2 * gamma(1+1/2) )**self.ndim * 1/(gamma(1+self.ndim/2))
            self.kernel = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2) )) if np.sum(x_n**2) <= 1 else 0
            self.kernel_der = lambda x_n: 1/C * np.exp(-1/(1-np.sum(x_n**2))) * 2*x_n/(-1 + np.sum(x_n**2))**2 if np.sum(x_n**2) <= 1 else 0
##            print('C',C)
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
            self.points_list = [[]]* self.num_id_funcs
            self.num_sample_list = [[]]* self.num_id_funcs
            self.points1d = np.linspace(-1,1,self.npop_stoch+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
            self.num_samples = self.points1d.size

            self.points_list[0] = np.reshape(np.linspace(-1,1,self.npop_stoch+2)[1:-1], (self.npop_stoch, 1))
            self.num_sample_list[0] = self.points_list[0].size

            for kk in range(1,self.num_id_funcs):
                if kk ==1:
                    r = np.linspace(0, 1, self.npop_stoch+2)[1:-1]
                    theta = np.linspace(0, 2*np.pi, self.npop_stoch, endpoint=False)
                    R, TH = np.meshgrid(r, theta, indexing='ij')

                    x = (R * np.cos(TH)).flatten()
                    y = (R * np.sin(TH)).flatten()

                    points_circle = np.vstack((x, y)).T

                    self.points_list[kk] = points_circle.copy()
                    self.num_sample_list[kk] = self.points_list[kk].shape[0]

                    self.Rvals = R.flatten()
                else:                
                    self.num_samples = self.pointsNd.size
                    pointsNd = np.zeros(( int(self.npop_stoch)**(kk+1),kk+1))
                    base_nd = np.linspace(-1,1,self.npop_stoch+2)[1:-1] ### values on the boundary are throwaway anyways due to rho = 0 and del rho = 0
                    coords = tuple([base_nd]*(kk+1))
                    mesh_nd = np.meshgrid(*coords, indexing = 'ij')
                    for i in range(kk+1):
                        pointsNd[:,i] = np.reshape(mesh_nd[i], int(self.npop_stoch)**(kk+1)) 
                    self.points_list[kk] = pointsNd.copy()
                    self.num_sample_list[kk] = self.points_list[kk].shape[0]
            
            
        elif self.npop > 4:


            r = np.linspace(0, 1, self.npop_stoch+2)[1:-1]
            theta = np.linspace(0, 2*np.pi, self.npop_stoch, endpoint=False)
            R, TH = np.meshgrid(r, theta, indexing='ij')

            x = (R * np.cos(TH)).flatten()
            y = (R * np.sin(TH)).flatten()

            points_circle = np.vstack((x, y)).T
            self.pointsNd = points_circle.copy()

            self.Rvals = R.flatten()
            
            self.num_samples = self.pointsNd.size
        return

    def generate_weights(self, number = 0):
        if number == 0: 
            if self.use_one_directional_smoothing:
                self.weights_list = [[]]* self.num_id_funcs
                self.weights_der_list = [[]]* self.num_id_funcs
                for kk in range(self.num_id_funcs):
                    if kk == 1:
                        self.weights_list[kk] = self.Rvals[:,None].T * np.asarray([self.kernel_list[kk](x) for x in self.points_list[kk]])
                        self.weights_der_list[kk] = self.Rvals[:,None] * np.asarray([self.kernel_der_list[kk](x) for x in self.points_list[kk]])
##                        print('%i -d integral approximation:' %(kk+1),2*np.pi/(int(self.npop_stoch)) * 1/(int(self.npop_stoch)+1) * np.sum(self.weights_list[kk]))
##                        print('%i -d der integral approximation:' %(kk+1), 2*np.pi/(int(self.npop_stoch)) * 1/(int(self.npop_stoch)+1) * np.sum(self.weights_der_list[kk]))
                    else:
                        self.weights_list[kk] = np.asarray([self.kernel_list[kk](x) for x in self.points_list[kk]])
                        self.weights_der_list[kk] = np.asarray([self.kernel_der_list[kk](x) for x in self.points_list[kk]])
##                        print('%i -d integral approximation:' %(kk+1),2**(kk+1)* 1/(int(self.npop_stoch)+1)**(kk+1) * np.sum(self.weights_list[kk]))
##                        print('%i -d der integral approximation:' %(kk+1),2**(kk+1) * 1/(int(self.npop_stoch)+1)**(kk+1) * np.sum(self.weights_der_list[kk]))
                

            elif self.npop > 4:

                
                self.weightsNd = self.Rvals[:,None].T *np.asarray([self.kernel(x) for x in self.pointsNd])
                self.weights_derNd = self.Rvals[:,None] * np.asarray([self.kernel_der(x) for x in self.pointsNd])

##                print('Nd integral approximation:',2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.sum(self.weightsNd))
##                print('Nd der integral approximation:',2*np.pi/(int(self.npop_stoch)) * 1/(int(self.npop_stoch)+1) * np.sum(self.weights_derNd))
        else:
            self.weights_list[number] = np.asarray([self.kernel_list[number](x) for x in self.points_list[number]])
            self.weights_der_list[number] = np.asarray([self.kernel_der_list[number](x) for x in self.points_list[number]])
            
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
            y = np.ones((self.npop,self.ndim))
            for i in range(self.ndim):
                y[:,i] = length * self.points1d * direction[i] + center[i]
            ### sample along line
            y = list(y)
        return y

    def ask_nd(self, center, lengths, *args):
        ### if gradient exists only return point evaluation
        if self.gradient_available:
            y =np.ones((self.npop,self.ndim))*center
            y = list(y)
            y.append(center)
        ### if no gradient available return sampling points in stoch region
        else:

            y = np.zeros((self.num_sample_list[len(lengths)-1], self.ndim ))
            direction_list = args[0]
            dir_mat = np.ones((self.ndim,len(lengths)))
            for kk in range(len(lengths)):
                dir_mat[:,kk] = lengths[kk] * direction_list[:,kk]

            transformed_points = (dir_mat @ self.points_list[len(lengths)-1].transpose())
            for kk in range(self.ndim):
                y[:,kk] = center[kk] + transformed_points[kk,:]
            y = list(y)
        return y
    

    def compute_val(self, inp, sigma_loc = None, func_vals = None, *args):
        ### computes the function value for smoothed functions
        
        if self.gradient_available:
            if func_vals is None:
                val = self.obj([inp,inp])[-1]
            else:
                val = func_vals[-1]
        elif self.use_one_directional_smoothing:
            if sigma_loc is None:
                ### get fun val, gradient val, min eigenvalue
                (f,df,E,dE) = self.obj_eigs([inp])
                active_id_funcs = []
                for kk in range(self.num_id_funcs):
                    if E[-1][kk]**2 < self.eps_val[kk]-self.tol*self.eps_val[kk]:
                        active_id_funcs.append(kk)

                directions = np.zeros((self.ndim,len(active_id_funcs)))
                sigma_loc_list = []
                if len(active_id_funcs) > 0:

                    cc = 0
                    for kk in active_id_funcs:
                        z = ((E[-1][kk])**2-self.kappa[kk])/(self.eps_val[kk] - self.kappa[kk])
                        if z >=0: 
                            sigma_loc_list.append( self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3) )
                        else:
                            sigma_loc_list.append(self.sigma)
                        directions[:,cc] = dE[-1][kk,:]/np.linalg.norm(dE[-1][kk,:])
                        cc+=1

                    if len(active_id_funcs) ==1:
                        y = self.ask_nd(inp, sigma_loc_list, directions)
                        f_y = self.obj(y)

                        val = 2 * 1/(self.npop_stoch+1) * np.inner(f_y, self.weights_list[0].T)

                    else:
                        y = self.ask_nd(inp, sigma_loc_list, directions)
                        f_y = self.obj(y)

                        val = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.inner(f_y, self.weights_list[len(active_id_funcs)-1])

                else:
                    val = self.obj([inp,inp])[-1] 
            else: 
                direction = args[0]
                if func_vals is None:
                    Gram = direction.copy().transpose() @ direction.copy()
                    Grank = np.linalg.matrix_rank(Gram)
                    if Grank < len(sigma_loc):
                        sortid = np.argsort(sigma_loc_list)
                        direction = direction[:,sortid[-1-len(sigma_loc)+Grank:]]
                        sigma_loc = np.asarray(sigma_loc)[sortid[-1-len(sigma_loc)+Grank:]].tolist()
                        
                    y = self.ask_nd(inp, sigma_loc, direction)
                    f_y = self.obj(y)
                else:
                    f_y = func_vals
                if len(sigma_loc) == 1:
                    val = 2**len(sigma_loc) * 1/(self.npop_stoch+1)**len(sigma_loc) * np.inner(f_y, self.weights_list[len(sigma_loc)-1].transpose())
                else:

                    val = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.inner(f_y, self.weights_list[len(sigma_loc)-1])

        else:

            y = self.ask(inp)
            f_y = self.obj(y)
            val = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.inner(f_y, self.weightsNd)
        return val

    def compute_grad(self, inp, function_values = None, gradient_values = None, sigma_loc = None, sigma_loc_der = None, *args):
        ### computes the gradient value for smoothed functions
        if self.gradient_available:
            if function_values is None:
                (vv,gradd) = self.obj_grad([inp,inp])
                grad = gradd[-1]
            else:
                grad = gradient_values[0]
                
        elif self.use_one_directional_smoothing:
            ### computes the local length of smoothing if none is given
            if sigma_loc is None:

                (f,df,E,dE) = self.obj_eigs([inp])
                active_id_funcs = []
                for kk in range(self.num_id_funcs):
                    if E[-1][kk]**2 < self.eps_val[kk]-self.tol*self.eps_val[kk]:
                        active_id_funcs.append(kk)
                directions = np.zeros((self.ndim,len(active_id_funcs)))
                sigma_loc_list = []

                if len(active_id_funcs) == 0:
                    (vv,gradd) = self.obj_grad([inp,inp])
                    grad = gradd[-1]
                    return grad
                sigma_loc_der_list = []
                cc = 0
                for kk in active_id_funcs:
                    z = ((E[-1][kk])**2-self.kappa[kk])/(self.eps_val[kk] - self.kappa[kk])
                    zder =  ((E[-1][kk])*2 * np.linalg.norm(dE[-1][kk,:]))/(self.eps_val[kk] - self.kappa[kk])
                    if z>= 0:
                        sigma_loc_list.append( self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3) )
                        sigma_loc_der_list.append( - self.sigma * zder * ( 30*(1-z)**4 - 60*(1-z)**3 + 30*(1-z)**2) )
                    else:
                        sigma_loc_list.append(self.sigma)
                        sigma_loc_der_list.append(0)
                    directions[:,cc] = dE[-1][kk,:]/np.linalg.norm(dE[-1][kk,:])
                    cc+=1

                y = self.ask_nd(inp, sigma_loc_list,directions)
                
                function_values = self.obj(y)
                (vv,gradient_vvalues) = self.obj_grad([inp,inp])
                gradient_values = gradient_vvalues[-1]

                if len(active_id_funcs) ==1:
                    grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), (self.points_list[0][1:] * self.weights_list[0][1:]).transpose() )
                    grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der_list[0].transpose())
                    grad_0 = ( sigma_loc_der_list[0]/sigma_loc_list[0] * grad_0_front + 1/sigma_loc_list[0] * grad_0_back)
                    if self.ndim ==2: 
                        perp_dir = np.array([directions[1], -directions[0]])/np.linalg.norm([directions[1], -directions[0]])
                        grad_2 = np.dot(gradient_values, perp_dir)[0]
                        grad = ( np.asarray([directions[0] * grad_0 + directions[1] * grad_2,
                                             directions[1] * grad_0 - directions[0] * grad_2]) )
                    else:
                        perp_dir_1 = np.array([directions[1], -directions[0],0])/np.linalg.norm([directions[1], -directions[0],0])
                        perp_dir_2 = np.array([0,0,1])/np.linalg.norm([0,0,1])
                        grad_21 = np.dot(gradient_values, perp_dir_1)[0]
                        grad_22 = np.dot(gradient_values, perp_dir_2)
                        grad = np.squeeze( np.asarray([directions[0] * grad_0 + directions[1] * grad_21,
                                         directions[1] * grad_0 - directions[0] * grad_21, grad_22 ]) )

                else:

                    Gram = directions.copy().transpose() @ directions.copy()
                    volume_el = np.sqrt(np.linalg.det(Gram))

                    grad_0_back = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.inner(function_values, self.weights_der_list[len(active_id_funcs)-1].transpose())
                    grad_0 = 1/np.asarray(sigma_loc_list) * grad_0_back 


                    direction_mat = np.zeros((self.ndim,self.ndim))
                    direction_mat[:,:len(active_id_funcs)] = directions.copy()
                    if self.ndim ==2: 
                        perp_dirs = null_space(directions)
                        for kk in range(self.ndim-len(active_id_funcs)):
                            direction_mat[:,len(active_id_funcs)+kk] = perp_dirs[kk].copy()/np.linalg.norm(perp_dirs[kk].copy())

                    else: 
                        perp_dirs = np.asarray([0,0,1])
                        direction_mat[:,-1] = perp_dirs.copy()/np.linalg.norm(perp_dirs.copy())
                    grad_vals = np.zeros((self.ndim,))
                    grad_vals[0:len(sigma_loc_list)] = grad_0
                    grad_vals[len(sigma_loc_list):] = perp_dirs.transpose() @ gradient_values.copy()

                    grad_vals = np.zeros((self.ndim,))
                    grad_vals[0:len(active_id_funcs)] = grad_0
                    grad_vals[len(active_id_funcs):] = perp_dirs.transpose() @ gradient_values.copy()

                    grad = gradient_values.copy()
                    grad = direction_mat @ grad_vals
                    
            
            else:
                directions = args[0]
                if len(sigma_loc) ==1:
                    grad_0_front = np.inner( ( function_values[:-1]-function_values[1:]), (self.points_list[0][1:] * self.weights_list[0][1:]).transpose() )
                    grad_0_back = 2 * 1/(self.npop_stoch+1) * np.inner(function_values, self.weights_der_list[0].transpose())
                    grad_0 = ( sigma_loc_der[0]/sigma_loc[0] * grad_0_front + 1/sigma_loc[0] * grad_0_back)[0]
                    if self.ndim == 2:
                        perp_dir = np.array([directions[1], -directions[0]])/np.linalg.norm([directions[1], -directions[0]])
                        grad_2 = np.dot(gradient_values, perp_dir)[0][0]
                        grad = np.squeeze( np.asarray([directions[0] * grad_0 + directions[1] * grad_2,
                                         directions[1] * grad_0 - directions[0] * grad_2]) )
                    else:
                        perp_dir_1 = np.array([directions[1], -directions[0],0])/np.linalg.norm([directions[1], -directions[0],0])
                        perp_dir_2 = np.array([0,0,1])/np.linalg.norm([0,0,1])
                        grad_21 = np.dot(gradient_values, perp_dir_1)[0][0]
                        grad_22 = np.dot(gradient_values, perp_dir_2)[0]
                        grad = np.squeeze( np.asarray([np.squeeze(directions[0] * grad_0) + np.squeeze(directions[1] * grad_21),
                                         np.squeeze(directions[1] * grad_0) - np.squeeze(directions[0] * grad_21), grad_22 ]) )
                else:
                    
                    grad_0_back = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.inner(function_values, self.weights_der_list[len(sigma_loc)-1].T)
                    grad_0 = 1/np.asarray(sigma_loc) * grad_0_back 
                    
                    direction_mat = np.zeros((self.ndim,self.ndim))
                    direction_mat[:,:len(sigma_loc)] = directions.copy()
                    perp_dirs = null_space(directions)
                    
                    if self.ndim ==2: 
                        perp_dirs = null_space(directions)
                        for kk in range(self.ndim-len(sigma_loc)):
                            direction_mat[:,len(sigma_loc)+kk] = perp_dirs[kk].copy()/np.linalg.norm(perp_dirs[kk].copy())
                    else: 
                        perp_dirs = np.asarray([0,0,1])
                        direction_mat[:,-1] = perp_dirs.copy()/np.linalg.norm(perp_dirs.copy())
                        
                    grad_vals = np.zeros((self.ndim,))
                    grad_vals[0:len(sigma_loc)] = grad_0
                    grad_vals[len(sigma_loc):] = perp_dirs.transpose() @ gradient_values[0][:].copy()
                    grad = direction_mat @ grad_vals


        else:
            if function_values is None:
                y = self.ask(inp)
                function_values = self.obj(y)
            grad = 2*np.pi/(int(self.npop_stoch))* 1/(int(self.npop_stoch)+1) * np.sqrt(1)/self.sigma * np.inner(function_values, self.weights_derNd.transpose())

        return grad

    def compute_smoothing_direction(self, inp, active_id_funcs):
        grad_direction = np.zeros((self.ndim,len(active_id_funcs)))
        self.eval_count += self.ndim
        
        ## compute the direction of the discontinuity from the gradient of the smallest eigenvalue
        for k in range(self.ndim):
            dir_1 = np.zeros((self.ndim)) ## x direction vector
            dir_1[k] = 1
            (ta,dta,E_fwd_1) = self.obj_eigs([inp+ dir_1*self.sten_size ])
            (ta,dta,E_bwd_1) = self.obj_eigs([inp - dir_1*self.sten_size ])
            grad_direction[k,:] = (E_fwd_1[0][active_id_funcs] - E_bwd_1[0][active_id_funcs])/(2*self.sten_size)
            
        return [grad_direction]

    def tell(self, solutions, function_values, gradient_values = None):
        ### solutions contains sampling points (2d, 1d, or point eval)
        ### function_values contains func vals at sampling points (2d, 1d, or point eval)
        ### gradient_values are necessary in 1d smoothing, give exact grad at current point
        ###     and let you decompose into reliable and unreliable part, then build a combi version
        self.eval_count = 0
        if not self.unchanged_mu:
            if self.use_one_directional_smoothing and not self.gradient_available:
                grad = self.compute_grad(self.mu,function_values, gradient_values, self.sigma_loc_mu_list,self.sigma_loc_mu_der_list, self.grad_dirs)
            else:
                grad = self.compute_grad(self.mu,function_values, gradient_values)
            if np.linalg.norm(grad) > self.aleph:
                grad = grad/np.linalg.norm(grad)
                step = - grad ### step direction
                aleph = self.aleph
                
            else:
                step = - grad ### step direction
                aleph = 1#self.aleph
            

            mu_aleph = self.mu.copy() + aleph * step
            m = np.inner(grad, step)
            self.history['gradients'].append(step)
            
            if np.linalg.norm(grad) < 10**-8 and not self.unchanged_mu:
                self.unchanged_mu = True#
                Armijo_iter = 1
            if self.Armijo and not self.unchanged_mu:
                if self.use_one_directional_smoothing and not self.gradient_available:
                    ff_mu = self.compute_val(self.mu, self.sigma_loc_mu_list, function_values, self.grad_dirs)
                else:
                    ff_mu = self.compute_val(self.mu)
                m = np.inner(grad, step)
                self.eval_count +=1
                Armijo_iter = 1
                ff_mu_try = self.compute_val(mu_aleph)
                while ( ff_mu_try > ff_mu + 10**-4 * aleph * m ) and not self.stop_algorithm:

                    aleph *= 0.5
                    mu_aleph = self.mu.copy() + aleph * step
                    Armijo_iter +=1
                    ff_mu_try = self.compute_val(mu_aleph)
                    self.eval_count += 1
                    if Armijo_iter > 50:
                        self.stop_algorithm = True
                        self.unchanged_mu = True
            self.mu = mu_aleph.copy()
            self.step_old = step.copy()
            self.history['alphas'].append(aleph)
            self.history['Armijo'].append(Armijo_iter)

        else:
            aleph = 0
            mu_aleph = self.mu.copy() 

    def result(self):
        return self.mu

    ### optimization outer shell
    def optimize(self, objective_function, iterations=1000, gradient_function = None, args=(), Armijo = True, eps_val = [2], kappa = [1], aleph = 0.3):
        ### objective function should return for a given vector of sampling points
        ###     a vector of function values f
        ###     a vector of gradient values df (if 1d smoothing or point eval scheme)
        ###     a vector of the minimal eigenvalue E (over all time points) (if 1d smoothing)
        
        iteration = 0
        if len(eps_val) == 1:
            self.eps_val = [eps_val[0]**2]*self.num_id_funcs
            self.kappa = [kappa[0]**2]*self.num_id_funcs
        elif len(eps_val) != self.num_id_funcs:
            print('eps_val must either be a list of length 1 or of length equal to number of zero set functions')
        else:
            for kk in range(self.num_id_funcs):
                 self.eps_val[kk] = eps_val[kk]**2
                 self.kappa[kk] = kappa[kk]**2
                 
        self.Armijo = Armijo
        self.aleph = aleph

        self.unchanged_mu = False
        
        self.obj = lambda x: objective_function(x)
        if self.gradient_available: 
            self.obj_grad = lambda x: objective_function(x, ['derivative'])
        if self.check_for_stability:
            self.obj_eigs = lambda x: objective_function(x, ['eigs', 'derivative'])

        self.sten_size = 10**-4 ### size of stencil  for approximation of eigenvalue derivative
        
        self.generate_samples()
        self.generate_weights()
##        while not self.stop_algorithm and iteration < iterations:
        while iteration < iterations:
            ### if stability check, check whether close to discontinuity (only in 1d smoothing used)
            if self.check_for_stability:
                ### get fun val, gradient val, min eigenvalue
                (fE,dfE,E,dE) = self.obj_eigs([self.mu])
                active_id_funcs = []
                for kk in range(self.num_id_funcs):
                    if E[-1][kk]**2 < self.eps_val[kk] -self.tol*self.eps_val[kk]:
                        active_id_funcs.append(kk)
                self.grad_dirs = np.zeros((self.ndim,len(active_id_funcs)))
                if len(active_id_funcs)>0: ### check func val if close to x = 0
                    unstable = True ### set unstable flag true, indicated smoothing needed
                    self.npop = self.npop_stoch**len(active_id_funcs) ### set number of sampling to smoothing number
                    self.gradient_available = False ### indicate you need approximate gradient from method
                    ## compute the direction of the discontinuity from the gradient of the smallest eigenvalue

                    ### choice between smoothing in 2d and 1d smoothing
                    if self.use_one_directional_smoothing: ### 1d smoothing
                        ### performs smoothing in direction of eigval gradient
                        self.sigma_loc_mu_list = []
                        self.sigma_loc_mu_der_list = []
                        cc = 0
                        for kk in active_id_funcs:
                            zder =  ((E[-1][kk])*2 * np.linalg.norm(dE[-1][kk,:]))/(self.eps_val[kk] - self.kappa[kk])
                            z = ((E[-1][kk])**2-self.kappa[kk])/(self.eps_val[kk] - self.kappa[kk])
                            if z >= 0:
                                self.sigma_loc_mu_list.append(self.sigma * ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3))
                                self.sigma_loc_mu_der_list.append( - self.sigma * zder * ( 30*(1-z)**4 - 60*(1-z)**3 + 30*(1-z)**2))
                            else:
                                self.sigma_loc_mu_list.append(self.sigma)
                                self.sigma_loc_mu_der_list.append(zder * 0)
                            self.grad_dirs[:,cc] = dE[-1][kk,:]/np.linalg.norm(dE[-1][kk,:])
                            cc += 1
                            
                        X = self.ask_nd(self.mu, self.sigma_loc_mu_list, self.grad_dirs)
                        self.history['smoothing_directions'].append(self.grad_dirs)
                    else: ### 2d smoothing
                        ### perform full smoothing
                        X = self.ask(self.mu)
                        f = self.objective_function(X)
                else:
                    ### if not unstable no smoothing
                    unstable = False
                    self.gradient_available = True
                    self.npop = self.npop_der
                    X = self.ask(self.mu)
            else:
                ### for general stoch methods, with no stab check
                X = self.ask(self.mu)

            
            ### generate gradient (or no gradient) depending method
            if self.gradient_available:
                f = [fE,fE]
                df = np.ones((self.ndim,self.ndim))*dfE
                self.tell(X, f, df)
            elif self.use_one_directional_smoothing:
                f = objective_function(X)
                df_1 = dfE
                self.tell(X, f, df_1)
            else:
                f = objective_function(X)
                self.tell(X, f)

                
                    
            ### save data of the optimization run (num iterations, num fun evals)
            if len(self.history['NumIters']) == 0:
                self.history['NumIters'].append(self.npop)
                self.history['FuncEvals'].append(self.npop)
            else: 
                self.history['NumIters'].append(self.history['NumIters'][-1] + self.eval_count*self.npop)
                self.history['FuncEvals'].append(self.npop)

            if self.check_for_stability and unstable:
                self.npop = self.npop_der
                unstable = False
                self.gradient_available = True
                

            self.history["solution"] += [self.mu.copy()]
            iteration += 1
        return self





