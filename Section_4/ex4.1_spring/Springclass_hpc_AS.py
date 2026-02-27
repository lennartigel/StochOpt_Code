import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig
from scipy.linalg import block_diag

class FractureProblem:
    def __init__(self, n_time_steps, end_time):
        self.k = 10.0  # spring constant
        self.G1 = 1.0  # fracture toughness of interface 1
        self.G2 = 1.0 # fracture toughness of interface 2
        self.c1 = 100.0  # elastcity constant for interfaces
        self.c2 = 100.0  # elastcity constant for interfaces

        self.n_time_steps = n_time_steps
        base = np.linspace(0.0, end_time, n_time_steps)
        self.T = base.copy()
        self.Tbackup = base.copy()## this backup is necessary to reset the time vector after an optimization for a new design

        self.sol = np.zeros((4, n_time_steps))

        self.smax = lambda a,b: (a+b+np.sqrt((a-b)**2 + 10**-12))/2
        self.smax_der = lambda a,b,dera,derb: (dera+derb + (a-b)*(dera-derb)/np.sqrt((a-b)**2 + 10**-12))/2
            
        
    def softmax(self,out1,out2):
        ### softmax function
        np.seterr(over='ignore')
        zz = np.maximum(out1,out2)
        softmaxalpha = 50
        
        ret = zz + 1/softmaxalpha * np.log( (
                                                    np.exp(softmaxalpha * (out1-zz)) + np.exp(softmaxalpha * (out2-zz))
                                                    )
                                           )
        return ret

    # total potential energy of the system
    def potential_energy(self, x, *args):
        u = x[:2]  # interfacial opening
        d = x[2:]  # damage
        u_d = args[0]  # dirichlet BC for displacement
        umax = args[1]
        E = (1*(
            # stored elastic energy in spring
            0.5 * self.k * (u_d - u[0] - u[1]) ** 2
            # stored elastic energy at interface 1
            + (1.0 - d[0]) ** 2 * 0.5 * self.c1 * max(umax[0]*umax[0],u[0] * u[0])
            ## stored elastic energy at interface 2
            + (1.0 - d[1]) ** 2 * 0.5 * self.c2 * max(umax[1]*umax[1],u[1] * u[1])
            ## dissipated fracture energy at interface 1
            + self.G1 * d[0] * d[0]
            ## dissipated fracture energy at interface 2
            + self.G2 * d[1] * d[1]
        ))
        return E
    
    # derivative of the total potential energy of the system
    ### this is mainly here to reference 
    def potential_energy_der(self,x,y,*args):
        u = x[:2]
        d = x[2:]
        ua = y[:2,:]
        da = y[2:,:]
        u_d = args[0]
        umax = args[1]
        umaxa = args[2]
        Ea = (1*(
             self.k * (u_d - u[0] - u[1]) * (-ua[0,:] - ua[1,:])
            + (1.0 - d[0]) ** 2 * self.c1 * max(umax[0]*umax[0],u[0] * u[0]) * umaxa[0] * ua[0,:]
            + (1.0 - d[1]) ** 2 * self.c2 * max(umax[1]*umax[1],u[1] * u[1]) * umaxa[0] * ua[1,:]
            - (1.0 - d[0]) * self.c1 * max(umax[0]*umax[0],u[0] * u[0]) * da[0,:]
            - (1.0 - d[1]) * self.c2 * max(umax[1]*umax[1],u[1] * u[1]) * da[1,:]
            + 2 *self.G1 * d[0] * da[0,:]
            + 2 *self.G2 * d[1] * da[1,:]
            ))
        return Ea
        
    
    def solve_u(self, x, *args):
        d = x[2:]  # damage
        u_d = args[0]  # dirichlet BC for displacement
        K = np.zeros((2, 2)) # system matrix to solve system for u
        K[0, 0] = self.k + self.c1 * (1.0 - d[0]) ** 2
        K[1, 1] = self.k + self.c2 * (1.0 - d[1]) ** 2
        K[0, 1] = self.k
        K[1, 0] = self.k

        rhs = self.k * u_d * np.ones(2)
        return np.linalg.inv(K).dot(rhs)

    def solve_du(self, x, y, *args):
        d = x[2:]
        u = x[:2]
        u_d = args[0]
        dd = y
        K = np.zeros((2, 2)) ### system matrix for sensitivity wrt u
        K2 = np.zeros((2,2)) ### system matrix for sensitivity wrt d
        ddu = np.ones((2,2))
        K[0, 0] = self.k + self.c1 * (1.0 - d[0]) ** 2
        K[1, 1] = self.k + self.c2 * (1.0 - d[1]) ** 2
        K[0, 1] = self.k
        K[1, 0] = self.k
        K2[0, 0] = 2* (d[0] - 1) * self.c1 * u[0]
        K2[1, 1] = 2* (d[1] - 1) * self.c2 * u[1]

        return -1*np.linalg.inv(K).dot(K2.dot(dd))

    def solve_d(self, x, *args):
        u = x[:2]  # damage
        u_max = args[1] ## irreversability
        d = np.zeros(2)
        d[0] = self.c1 * self.smax(u[0]**2,u_max[0]**2) / (2.0 * self.G1 + self.c1 * self.smax(u[0]**2,u_max[0]**2) )
        d[1] = self.c2 * self.smax(u[1]**2,u_max[1]**2) / (2.0 * self.G2 + self.c2 * self.smax(u[1]**2,u_max[1]**2) )
        return d

    def solve_dd(self, x, y, *args):
        u = x[:2]  # damage
        d = x[2:]
        ddu = y
        umax = args[1]
        ddumax = args[2] ### tracks the history variable derivative
        dd = np.ones((2,2))
        L = np.zeros((2,2)) ### system matrix for sensitivity wrt d
        L2 = np.zeros((2,2)) ### system matrix for sensitivity wrt u
        L3 = np.zeros((2,2)) ### system matrix for sensitivity wrt umax
        L[0,0] = 2*d[0]
        L[1,1] = 2*d[1]
        L2[0,0] = 2* self.G1 + self.c1 * self.smax(u[0]**2, umax[0]**2)
        L2[1,1] = 2* self.G2 + self.c2 * self.smax(u[1]**2, umax[1]**2)

        L3[0,0] = 2* self.smax(u[0], umax[0]) * self.c1 * (d[0]-1)
        L3[1,1] = 2* self.smax(u[1], umax[1]) * self.c2 * (d[1]-1)
        return -1*np.linalg.inv(L2).dot(L + L3.dot(ddumax))

    def solve_u2(self, x, *args):
        d = x[2:]  # damage
        u_d = args[0]  # dirichlet BC for displacement
        u = np.zeros(2)
        u[0] = ( (1-d[1])**2 * self.c2/self.k * u_d )/( self.c1 * self.c2/self.k**2 * (1-d[0])**2 * (1-d[1])**2 + self.c1/self.k * (1-d[0])**2 + self.c2/self.k * (1-d[1])**2)
        u[1] = ( (1-d[0])**2 * self.c1/self.k * u_d )/( self.c1 * self.c2/self.k**2 * (1-d[0])**2 * (1-d[1])**2 + self.c1/self.k * (1-d[0])**2 + self.c2/self.k * (1-d[1])**2) 
        return u

    def solve_u2_der(self, x, *args):
        d = x[2:]  # damage
        t = args[0]  # dirichlet BC for displacement
        u = np.zeros(2)
        A0 = self.c1 /self.k * (1- d[0])**2
        A1 = self.c2 /self.k * (1- d[1])**2
        der_A0 = 2 * self.c1/self.k * (d[0]-1)
        der_A1 = 2 * self.c2/self.k * (d[1]-1)
        denom = (A0 * A1 + A0 + A1)
        u_der = np.zeros((2,2))
        u_der[0,0] = - ( A1*(A1 + 1) * der_A0* t ) /denom**2
        u_der[1,1] = - ( A0*(A0 + 1) * der_A1* t ) /denom**2
        u_der[0,1] = (der_A1 * t) / denom - ( A1 * der_A1 * ( A0 + 1) * t ) /denom**2
        u_der[1,0] = (der_A0 * t) / denom - ( A0 * der_A0 * ( A1 + 1) * t ) /denom**2
        return u_der
    
    def solve_u2_hess(self, x, *args):
        d = x[2:]  # damage
        t = args[0]  # dirichlet BC for displacement
        u = np.zeros(2)
        A0 = self.c1 /self.k * (1- d[0])**2
        A1 = self.c2 /self.k * (1- d[1])**2
        der_A0 = 2 * self.c1/self.k * (d[0]-1)
        der_A1 = 2 * self.c2/self.k * (d[1]-1)
        hess_A0 = 2 * self.c1/self.k
        hess_A1 = 2 * self.c2/self.k
        denom = (A0 * A1 + A0 + A1)
        u_hess = np.zeros((2,2,2))
        u_hess[0,0,0] = - ( A1 * (A1 + 1) * hess_A0 * t ) /denom**2 + 2 * ( (A1 + 1)**2 * A1 * der_A0**2 * t ) /denom**3
        u_hess[0,0,1] = - ( (2 * A1 + 1) * der_A1 * der_A0 * t )/ denom**2 + 2 * ( (A1 + 1) * A1 * der_A0* t * der_A1 * (A0 + 1)  ) /denom**3
        u_hess[0,1,0] = - ( (2 * A1 + 1) * der_A1 * der_A0 * t )/ denom**2 + 2 * ( (A0 + 1) * der_A1 * ( A1 + 1) * A1 * der_A0 * t ) /denom**3
        u_hess[0,1,1] = ( (hess_A1 * t) / denom - (der_A1**2 * t * (A0 + 1)) / denom**2
                          - ( ( der_A1**2 + A1 * hess_A1) * (A0 + 1) * t ) /denom**2 + 2 * ( A1 * der_A1**2 * (A0 + 1)**2 * t  ) /denom**3 )
        u_hess[1,0,0] = ( (hess_A0 * t) / denom - (der_A0**2 * t * (A1 + 1)) / denom**2
                          - ( ( der_A0**2 + A0 * hess_A0) * (A1 + 1) * t ) /denom**2 + 2 * ( A0 * der_A0**2 * (A1 + 1)**2 * t  ) /denom**3 )
        u_hess[1,0,1] = - ( (2 * A0 + 1) * der_A0 * der_A1 * t )/ denom**2 + 2 * ( (A1 + 1) * der_A0 * ( A0 + 1) * A0 * der_A1 * t ) /denom**3
        u_hess[1,1,0] = - ( (2 * A0 + 1) * der_A0 * der_A1 * t )/ denom**2 + 2 * ( (A0 + 1) * A0 * der_A1* t * der_A0 * (A1 + 1)  ) /denom**3
        u_hess[1,1,1] = - ( A0 * (A0 + 1) * hess_A1 * t ) /denom**2 + 2 * ( (A0 + 1)**2 * A0 * der_A1**2 * t ) /denom**3
        return u_hess
    
    def der_potential_energy_dd(self, d, *args):
        t = args[0]  # dirichlet BC for displacement
        xx = np.zeros((4,))
        xx[2:] = d
        u = self.solve_u2(xx, t)
        der_u = self.solve_u2_der(xx, t)
        out = np.zeros((2,))
        out[0] = ( - self.k * (t - u[0] - u[1]) * (der_u[0,0] + der_u[1,0])
                   + self.c1 * u[0] * der_u[0,0] * ( 1 - d[0] )**2 + self.c1 * ( d[0] - 1 ) * u[0]**2
                   + self.c2 * u[1] * der_u[1,0] * ( 1 - d[1] )**2
                   + 2 * self.G1 * d[0] )
        out[1] = ( - self.k * (t - u[0] - u[1]) * (der_u[0,1] + der_u[1,1])
                   + self.c1 * u[0] * der_u[0,1] * ( 1 - d[0] )**2 
                   + self.c2 * u[1] * der_u[1,1] * ( 1 - d[1] )**2 + self.c2 * ( d[1] - 1 ) * u[1]**2
                   + 2 * self.G2 * d[1] )
        return out

    def hess_potential_energy_dd(self, d, *args):
        t = args[0]  # dirichlet BC for displacement
        xx = np.zeros((4,))
        xx[2:] = d
        u = self.solve_u2(xx, t)
        der_u = self.solve_u2_der(xx, t)
        hess_u = self.solve_u2_hess(xx, t)
        out_hess = np.zeros((2,2))
        out_hess[0,0] = ( - self.k * (t - u[0] - u[1]) * (hess_u[0,0,0] + hess_u[1,0,0])
                          + self.k * (der_u[0,0] + der_u[1,0])**2
                          + self.c1 * ( u[0] * hess_u[0,0,0] + der_u[0,0]**2) * ( 1 - d[0] )**2
                          + 2 * self.c1 * u[0] * der_u[0,0] * ( d[0] - 1 )
                          + self.c1 * u[0]**2
                          + 2 * self.c1 * ( d[0] - 1 ) * u[0] * der_u[0,0]
                          + self.c2 * ( der_u[1,0]**2 + u[1] * hess_u[1,0,0] ) * ( 1 - d[1] )**2
                          + 2 * self.G1 )
        out_hess[0,1] = ( - self.k * (t - u[0] - u[1]) * (hess_u[0,0,1] + hess_u[1,0,1])
                          + self.k * (der_u[0,0] + der_u[1,0]) * (der_u[0,1] + der_u[1,1])
                          + self.c1 * ( u[0] * hess_u[0,0,1] + der_u[0,0] * der_u[0,1]) * ( 1 - d[0] )**2
                          + 2 * self.c1 * ( d[0] - 1 ) * u[0] * der_u[0,1]
                          + self.c2 * ( der_u[1,0]*der_u[1,1] + u[1] * hess_u[1,0,1] ) * ( 1 - d[1] )**2
                          + 2 * self.c2 * u[1] * der_u[1,0] * ( d[1] - 1 )
                          )
        out_hess[1,0] = ( - self.k * (t - u[0] - u[1]) * (hess_u[0,1,0] + hess_u[1,1,0])
                          + self.k * (der_u[0,1] + der_u[1,1]) * (der_u[0,0] + der_u[1,0])
                          + self.c1 * (u[0] * hess_u[0,1,0] + der_u[0,1] * der_u[0,0])* ( 1 - d[0] )**2
                          + 2 * self.c1 * u[0] * der_u[0,1] * ( d[0] - 1 )
                          + self.c2 * ( u[1] * hess_u[1,1,0] + der_u[1,0] * der_u[1,1] ) * ( 1 - d[1] )**2
                          + 2 * self.c2 * ( d[1] - 1 ) * u[1] * der_u[1,0]
                          )
        out_hess[1,1] = ( - self.k * (t - u[0] - u[1]) * (hess_u[0,1,1] + hess_u[1,1,1])
                          + self.k * (der_u[0,1] + der_u[1,1])**2
                          + self.c1 * (u[0] * hess_u[0,1,1] + der_u[0,1]**2 ) * ( 1 - d[0] )**2
                          + self.c2 * (u[1] * hess_u[1,1,1] + der_u[1,1]**2 ) * ( 1 - d[1] )**2
                          + 2 * self.c2 * ( d[1] - 1 ) * u[1] * der_u[1,1]
                          + self.c2 * u[1]**2
                          + 2 * self.c2 * ( d[1] - 1 ) * u[1] * der_u[1,1]
                          + 2 * self.G2 )
        return out_hess

    def solve_eigen(self,x,tj,*args):
        if len(args) >0:
            ## entries which are considered in the Hessian, entries are omitted if irreversibility was enforced on them
            ind = [0,1]
            ind+= args[0]
        else:
            ind = [0,1,2,3]
        ind = np.ix_(ind,ind)
        d = x[2:]
        u = x[:2]
        H = np.zeros((4,4))
        Huu = np.zeros((2,2))
        H[0,0] = self.k + self.c1 * (1-d[0])**2
        H[1,1] = self.k + self.c2 * (1-d[1])**2
        H[2,2] = 2*self.G1 + self.c1 * u[0]**2
        H[3,3] = 2*self.G2 + self.c2 * u[1]**2
        H[1,0] = self.k
        H[2,0] = 2 * self.c1 * (1-d[0]) * u[0]
        H[3,1] = 2 * self.c2 * (1-d[1]) * u[1]
        H[0,1] = H[1,0]
        H[0,2] = H[2,0]
        H[1,3] = H[3,1]
        (res,vec) = eig(H[ind])
        min_ind = np.argsort(np.real(res))
        ress = np.real(res)[min_ind]
        vecs = vec[min_ind]

        return (ress[1],vecs[1,:])
        

    def solve(self, method):
        self.temp_time_steps = self.n_time_steps ## initalize number of timesteps
        self.T = self.Tbackup ## initialize time
        x = np.zeros((4, self.n_time_steps))## solution vector
        y = np.zeros((1, self.n_time_steps))## eigenvalues
        z = np.zeros((4, 2))## solution derivatives
        ww = np.zeros((4,self.n_time_steps))## Eigenvectors
        (y[:,0],ww[:,0]) = self.solve_eigen(x[:,0],self.T[0])## prelimiary computation for the 0th timestep to compute the eigenvalue/vectors
        First_time = True## hasnt entered adaptive time refinement once
        i = 1
        umax = np.zeros(x[:2,:].shape) ## initialize tracking variable of maximum u value
        dumax = np.zeros((2, 2))## initialize tracking variable for derivaitve of umax variable
        dumax_1= np.zeros((2, 2)) ## history of umax derivative at prev timestep
        while i < self.temp_time_steps:

            if method == "staggered_one_step":
                x[:, i] = x[:, i - 1]
                x[:2, i] = self.solve_u(x[:, i], self.T[i])
                x[2:, i] = self.solve_d(x[:, i], self.T[i])
                (y[:,i],v) = self.solve_eigen(x[:,i])
                
                
            elif method == "staggered_multi_step":
                ### change this value depending on what you want to simulate
                Irrev = True
##                Irrev = False  ### comment/uncomment depending on need
                x[:, i] = x[:, i - 1]
                xd = x[2:, i].copy()
                xd2 = x[2:, i].copy()
                xu = x[:2, i].copy()
                zd = z[2:,:].copy()
                err = 1
                inner_iter = 1
                ### alternating stepping while error too large
                while err > 10**-6 and inner_iter < 100:
                    x[:2, i] = self.solve_u(x[:, i], self.T[i]) ### solve displacement
                    z[:2,:] = self.solve_du(x[:, i], z[2:,:], self.T[i])### solve displacement derivative
                    if Irrev: ### enforce irreversability of damage
                        utol= 10**-8
                        ####### original 
                        marker = (np.where(umax[:,i-1] > x[:2,i] +utol)[0]+2).tolist()
                        marker2 = (np.where(umax[:,i-1] > x[:2,i]+utol )[0]).tolist()       
                        umax[0,i] = self.smax(x[0,i],umax[0,i-1])
                        umax[1,i] = self.smax(x[1,i],umax[1,i-1])
                        dumax_1 = dumax.copy()
                        dumax[0] = self.smax_der(x[0,i],umax[0,i-1], z[0,:],dumax[0,:])
                        dumax[1] = self.smax_der(x[1,i],umax[1,i-1], z[1,:],dumax[1,:])
                        if len(marker) > 0 and False:
                            x[marker,i] = xd2[marker2].copy() 
                            z[marker,:] = zd[marker2,:].copy() 
                    x[2:, i] = self.solve_d(x[:, i], self.T[i], umax[:,i])### solve damage
                    z[2:,:] = self.solve_dd(x[:, i], z[:2,:], self.T[i], umax[:,i], dumax)### solve damage derivative
                    err = max([abs(x[0,i] - xu[0]), abs(x[1,i] -xu[1]), abs(x[2,i] - xd[0]), abs(x[3,i] - xd[1])])
                    xd = np.copy(x[2:, i])
                    xu = np.copy(x[:2, i])
                    zd = np.copy(z[2:, :])

                    inner_iter +=1
                    
                h = 1/10
                if Irrev:
                    if len(marker) > 0: ## irreversibiity violated somewhere
                        (y[:,i],w) = self.solve_eigen(x[:,i], self.T[i], marker)
                    else:
                        (y[:,i],w) = self.solve_eigen(x[:,i], self.T[i])
                else:
                    (y[:,i],v) = self.solve_eigen(x[:,i], self.T[i])
                    w = v
                ### scheme for the refinement at point of instability
                if y[:,i] > y[:,i-1] and First_time:
                    ## if eigenvalue jumps up, erase last 2 computed entries (i-1 & i) and put num_add many additional timesteps between them and recompute
                    ## in case i = 1, only the the eigenvalues for i > 0 get recomputed, since for i = 0 the computation is outside the loop and the
                    ## index i-2 would loop around the vector to -1
                    ## i -2 i -1 i i +1 i +2 becomes i -1 i -1 j_1 j_2 ... j_numd-1 i i +1 
                    First_time = False
                    num_add = 500 ## number of adaptive timesteps
                    x[:,i-1:i+1] = np.zeros((4,2))
                    umax[:,i-1:i+1] = np.zeros((2,2))
                    dumax = dumax_1.copy()
                    y[:,max(1,i-1):i+1] = np.zeros((1,2+min(0,i-2)))
                    ww[:,max(1,i-1):i+1] = np.zeros((4,2+min(0,i-2)))
                    
                    x = np.append(x,np.zeros((4,num_add)),axis = 1)
                    y = np.append(y,np.zeros((1,num_add)),axis = 1)
                    ww = np.append(ww,np.zeros((4,num_add)),axis = 1)
                    umax = np.append(umax,np.zeros((2,num_add)),axis = 1)
                    
                    backlog = self.T[:max(0,i-2)]## time points that came before i-1
                    frontlog = self.T[i+1:]## time points that will came after i-1
                    entrylog = np.linspace(self.T[max(0,i-2)], self.T[i], num_add+3 + min(0,i-2)) ## time points that get pushed between i -1 and i
                    self.T = np.append(np.append(backlog,entrylog), frontlog) ##new time vector
                    self.temp_time_steps += num_add ## increase number of timesteps
                    i -=2 +min(0,i-2) ## go one step back 

                i += 1 ## next step
                
            else:
                print("method not implemented")
        self.sol = x
        self.eigs = y
        self.eigs_der = ww
        self.der = z
        self.umax = umax[:,-1]
        self.umaxder = dumax

