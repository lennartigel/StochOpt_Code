import numpy as np
import os as os
import scipy.optimize as spopt
import datetime
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres as spgmres
from scipy.linalg import eigh
import scipy.sparse as spp
from numpy import exp
import time as time


class SpringFracture:
    def __init__(self, L, B, n_time_steps, ntnn = False, ref_fact = 5, endtime = 1):
        ### base parameters
        self.pref_dist = 1
        self.L = L
        self.B = B

        ### default value whether to print updates on progress
        self.print_progress = False
        self.save_output = True

        ### time points
        self.n_time_steps = n_time_steps
        self.n_time_steps_og = n_time_steps
        self.ref_fact = ref_fact
        self.n_time_steps_fine = self.ref_fact* n_time_steps
        self.basetime = np.linspace(0,1,n_time_steps)
        self.finetime = np.linspace(0,1,self.ref_fact* n_time_steps) ### we swtich to finer time when necessary
        self.T = self.finetime.copy()

        #### number of eigenvalues extra
        self.num_eigs = 1

        ### solution list
        self.Y = np.zeros((self.L*self.B*2,self.n_time_steps_fine))

        ## variables for edge computation
        self.num_edges = self.L*(self.B-1) + self.B*(self.L-1) + 1*(self.L-1)*(self.B-1)
        self.s_1 = self.L*(self.B-1)
        self.s_2 = self.s_1 + self.B*(self.L-1)
        
        ### cutoff parameters for damage variable
        self.R_1 = 1.2*self.pref_dist * np.ones((self.num_edges,))
        self.R_2 = self.R_1.copy() *2.5 -self.pref_dist* np.ones((self.num_edges,))


        
        self.pots = np.ones((self.num_edges,))*-1

        self.standard_str = 1.2

        ### viscosity parameter
        self.nu = 0.5

        ### default dissipation type
        self.diss_type = 'KV'

        ### tolerance for multistep
        self.tol = 10**-10

        ### boundary conditions
        self.boundary = ['right','left'] ### denotes on what boundaries the load is applied
        ### examples for boundary conditions
        drawdist = 2.5*1.2*self.pref_dist - self.pref_dist
        angle = np.pi/8
        r_0 = lambda x: drawdist * np.sin(np.pi/2 * x) * np.cos(angle)
        r_1 = lambda x: drawdist * np.sin(np.pi/2 * x) * np.sin(angle)
        l_0 = lambda x: 0
        l_1 = lambda x: 0
        self.loading = [[r_0,r_1],[l_0,l_1]] ### list of loading functions
        self.dim = [[0,1],[0,1]]
        self.loading_spatial_dep = False

        self.ntnn = ntnn
        if self.ntnn:
            self.num_edges_ntnn = (self.L-1) * (2*self.B - 3) + (self.L-2) * (self.B)
            self.s_1_ntnn = (self.L-2)*(self.B)
            self.R_1_ntnn = 1.15 * self.pref_dist * np.ones((self.num_edges_ntnn,))
            self.R_2_ntnn = 2.5 * self.pref_dist * np.ones((self.num_edges_ntnn,))
            self.pots_ntnn = np.ones((self.num_edges_ntnn,))*-1
            self.ntnn_weight = 1/4


        self.inhom_material = False ### check whether
        self.compute_mat_der = True
        if self.compute_mat_der:
            self.chevron = False
            self.topdown = False
            self.leftright = False
            self.circles = False
            self.Friedrich_setup = True
            self.Friedrich_setup_2 = False
            self.alpha_list = [1.2,1.2]
            if self.Friedrich_setup or self.Friedrich_setup_2: 
                self.alpha_fix_list = [1.2,1.2,1.2,1.2]
            self.num_alpha = 2
            self.Y_alpha = np.zeros((self.L*self.B*2,self.num_alpha, self.n_time_steps_fine))

        ### softmax parameters for either softmax functions
        self.softmaxalpha = 5
        self.steepness = 1/1000
        self.phi_max_ex = True
        self.smoothmaxalpha = 10**-12

        self.smax = lambda a,b: (a+b+np.sqrt((a-b)**2 + 10**-12))/2
        self.smax_der = lambda a,b,dera,derb: (dera+derb + (a-b)*(dera-derb)/np.sqrt((a-b)**2 + 10**-12))/2

        ### storage container
        self.id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        
        self.track_eigs = True


    def get_vertices(self):
        X = np.zeros((self.L,self.B,2))
        ### x coordinate values on odd rows
        X[:,:,0] = np.linspace(0,(self.B-1)*self.pref_dist, self.B) + self.pref_dist/2
        ### x coordinate values on even rows
        X[0::2,:,0] = np.linspace(0,(self.B-1)*self.pref_dist, self.B)
        ### y coordinate values
        X[:,:,1] = np.repeat(np.arange(self.L)*self.pref_dist*np.sqrt(3)/2,self.B).reshape(self.L,self.B)
        return X

    def resort(self,x):
        l = self.B * self.L
        z = np.reshape(x,(l,2))
        y = np.zeros(z.shape)
        for i in range(self.L): 
            y[self.B*i:self.B*(i+1),:] = z[i::self.B,:]
        return y
        
    def get_edges(self,vert, ret_index = False):
        edge_index = np.zeros((self.L * self.B, self.num_edges))
        edges = np.zeros((self.num_edges,2,2))

        rows_ev = int(self.L/2)
        rows_unev = int((self.L-1)/2)
        num_fwd_diags_even = rows_ev * (self.B)
        num_bwd_diags_even = rows_ev * (self.B-1)
        num_fwd_diags_uneven = rows_unev * (self.B-1)
        num_bwd_diags_uneven = rows_unev * (self.B)
        num_diags_even = rows_ev * (2 * self.B -1)
        num_fwd_diags_uneven = rows_unev * (self.B - 1)
        s2 = self.s_1 + num_fwd_diags_even
        s3 = num_fwd_diags_uneven + num_diags_even
        s4 = num_bwd_diags_even + s2
        
        ### horizontal
        for i in range(self.L):
            edges[i*(self.B-1):(i+1)*(self.B-1),0,:] = vert[i*self.B:i*self.B+self.B-1,:]
            edges[i*(self.B-1):(i+1)*(self.B-1),1,:] = vert[i*self.B+1:(i+1)*self.B,:]
            if ret_index:
                edge_index[range(i*self.B,i*self.B+self.B-1), range(i*(self.B-1),(i+1)*(self.B-1))] = 1
                edge_index[range(i*self.B+1,(i+1)*self.B), range(i*(self.B-1),(i+1)*(self.B-1))] = 2
        ### vertical
        for i in range(self.L-1):
            ### diagonal fwd on even (1)
            if i%2 == 0:
                edges[self.s_1 + self.B*int(i/2):self.s_1 + self.B*(int(i/2)+1),0,:] = vert[i*self.B:(i+1)*self.B,:]
                edges[self.s_1 + self.B*int(i/2):self.s_1 + self.B*(int(i/2)+1),1,:] = vert[(i+1)*self.B:(i+2)*self.B,:]
                if ret_index:
                    edge_index[range(i*self.B,(i+1)*self.B), range(self.s_1 + self.B*int(i/2),self.s_1 + self.B*(int(i/2)+1))] = 1
                    edge_index[range((i+1)*self.B,(i+2)*self.B), range(self.s_1 + self.B*int(i/2),self.s_1 + self.B*(int(i/2)+1))] = 2
            ### diagonal bwd on uneven (4)
            else:
                edges[self.s_1 + s3 + self.B*int((i-1)/2):self.s_1 + s3 +self.B*(int((i-1)/2)+1),0,:] = vert[i*self.B:(i+1)*self.B,:]
                edges[self.s_1 + s3 + self.B*int((i-1)/2):self.s_1 + s3 +self.B*(int((i-1)/2)+1),1,:] = vert[(i+1)*self.B:(i+2)*self.B,:]
                if ret_index:
                    edge_index[range(i*self.B,(i+1)*self.B), range(self.s_1 + s3 + self.B*int((i-1)/2),self.s_1 + s3 +self.B*(int((i-1)/2)+1))] = 1
                    edge_index[range((i+1)*self.B,(i+2)*self.B), range(self.s_1 + s3 + self.B*int((i-1)/2),self.s_1 + s3 +self.B*(int((i-1)/2)+1))] = 2
            
            ### diagonal bwd on even (3)
            if i%2 == 0: 
                edges[s4 + int(i/2)*(self.B-1):s4 + (int(i/2)+1)*(self.B-1),0,:] = vert[i*self.B+1:(i+1)*self.B,:]
                edges[s4 + int(i/2)*(self.B-1):s4 + (int(i/2)+1)*(self.B-1),1,:] = vert[(i+1)*self.B:(i+1)*self.B+self.B-1,:]
                if ret_index: 
                    edge_index[range(i*self.B+1,(i+1)*self.B), range(s4 + int(i/2)*(self.B-1),s4 + (int(i/2)+1)*(self.B-1))] = 1
                    edge_index[range((i+1)*self.B,(i+1)*self.B+self.B-1), range(s4 + int(i/2)*(self.B-1),s4 + (int(i/2)+1)*(self.B-1))] = 2
            ### diagonal fwd on uneven (2)
            else: 
                edges[s2 + int((i-1)/2)*(self.B-1):s2 + (int((i-1)/2)+1)*(self.B-1),0,:] = vert[i*self.B:i*self.B+self.B-1,:]
                edges[s2 + int((i-1)/2)*(self.B-1):s2 + (int((i-1)/2)+1)*(self.B-1),1,:] = vert[(i+1)*self.B+1:(i+2)*self.B,:]
                if ret_index: 
                    edge_index[range(i*self.B,i*self.B+self.B-1), range(s2 + int((i-1)/2)*(self.B-1),s2 + (int((i-1)/2)+1)*(self.B-1))] = 1
                    edge_index[range((i+1)*self.B+1,(i+2)*self.B), range(s2 + int((i-1)/2)*(self.B-1),s2 + (int((i-1)/2)+1)*(self.B-1))] = 2
        if ret_index:
            return edges, edge_index
        else:
            return edges

    def get_distance(self,edges):
        dist = np.sum((edges[:,0,:]- edges[:,1,:])**2,1)
        return dist

    def get_distance_der(self,edges, edges_der):
        dist = 2* np.sum((edges[:,0,:]- edges[:,1,:])  * (edges_der[:,0,:]- edges_der[:,1,:]),1)
        return dist
    
    def get_edges_ntnn(self,vert,ret_index = False):
        ntnn_edges = np.zeros((self.num_edges_ntnn,2,2))
        edge_index = np.zeros((self.L * self.B, self.num_edges_ntnn))
        for k in range(self.L-2):
            ### vertical edges
            ntnn_edges[k*(self.B):(k+1)*(self.B),0,:] = vert[k*self.B:(k+1)*self.B,:]
            ntnn_edges[k*(self.B):(k+1)*(self.B),1,:] = vert[(k+2)*self.B:(k+3)*self.B,:]
            if ret_index:
                edge_index[range(k*self.B,(k+1)*self.B), range(k*(self.B),(k+1)*(self.B))] = 1
                edge_index[range((k+2)*self.B,(k+3)*self.B), range(k*(self.B),(k+1)*(self.B))] = 2
                
        for k in range(self.L-1):
            ### left and right leaning edges
            ### even rows
            if k%2 == 0:
                ### left (1)
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3): self.s_1_ntnn + k*(2*self.B-3) + (self.B-2),0,:] = vert[k*self.B+2:(k+1)*self.B]
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3): self.s_1_ntnn + k*(2*self.B-3) + (self.B-2),1,:] = vert[(k+1)*self.B:(k+2)*self.B-2]
                ### right (3)
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3) + (self.B-2): self.s_1_ntnn + (k+1)*(2*self.B-3),0,:] = vert[k*self.B: (k+1)*self.B-1]
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3) + (self.B-2): self.s_1_ntnn + (k+1)*(2*self.B-3),1,:] = vert[(k+1)*self.B+1: (k+2)*self.B]
                if ret_index:
                    edge_index[range(k*self.B+2,(k+1)*self.B), range(self.s_1_ntnn + k*(2*self.B-3), self.s_1_ntnn + k*(2*self.B-3) + (self.B-2))] = 1
                    edge_index[range((k+1)*self.B,(k+2)*self.B-2), range(self.s_1_ntnn + k*(2*self.B-3), self.s_1_ntnn + k*(2*self.B-3) + (self.B-2))] = 2

                    edge_index[range(k*self.B,(k+1)*self.B-1), range(self.s_1_ntnn + k*(2*self.B-3) + (self.B-2), self.s_1_ntnn + (k+1)*(2*self.B-3))] = 1
                    edge_index[range((k+1)*self.B+1,(k+2)*self.B), range(self.s_1_ntnn + k*(2*self.B-3) + (self.B-2), self.s_1_ntnn + (k+1)*(2*self.B-3))] = 2

            else:
                ### left (2)
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3): self.s_1_ntnn + k*(2*self.B-3) + (self.B-1),0,:] = vert[k*self.B+1:(k+1)*self.B]
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3): self.s_1_ntnn + k*(2*self.B-3) + (self.B-1),1,:] = vert[(k+1)*self.B:(k+2)*self.B-1]
                ### right (4)
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3) + (self.B-1): self.s_1_ntnn + (k+1)*(2*self.B-3),0,:] = vert[k*self.B: (k+1)*self.B-2]
                ntnn_edges[self.s_1_ntnn + k*(2*self.B-3) + (self.B-1): self.s_1_ntnn + (k+1)*(2*self.B-3),1,:] = vert[(k+1)*self.B+2: (k+2)*self.B]

                if ret_index:
                    edge_index[range(k*self.B+1,(k+1)*self.B), range(self.s_1_ntnn + k*(2*self.B-3), self.s_1_ntnn + k*(2*self.B-3) + (self.B-1))] = 1
                    edge_index[range((k+1)*self.B,(k+2)*self.B-1), range(self.s_1_ntnn + k*(2*self.B-3), self.s_1_ntnn + k*(2*self.B-3) + (self.B-1))] = 2

                    edge_index[range(k*self.B,(k+1)*self.B-2), range(self.s_1_ntnn + k*(2*self.B-3) + (self.B-1), self.s_1_ntnn + (k+1)*(2*self.B-3))] = 1
                    edge_index[range((k+1)*self.B+2,(k+2)*self.B), range(self.s_1_ntnn + k*(2*self.B-3) + (self.B-1), self.s_1_ntnn + (k+1)*(2*self.B-3))] = 2
                    
        if ret_index:
            return ntnn_edges, edge_index
        else: 
            return ntnn_edges
    
    def damage_var(self,r):
        np.seterr(divide='ignore')
        phi = np.zeros(r.shape)


        z = (r-self.R_1**2)/(self.R_2**2 - self.R_1**2)

        fz = ( 6*(z)**5 - 15*(z)**4 + 10*(z)**3)
        fz = 3*z**2 - 2*z**3
        fz = (1 - np.cos(np.pi*z) )/2

        phi = np.where(z <= 0, 0, fz)
        phi = np.where(z >= 1, 1, phi)
        
        
        return phi

    def damage_var_mat(self,r,r_mat):
        np.seterr(divide='ignore',invalid='ignore')
        phi_alpha = np.zeros(r_mat.shape)
        expansion = np.ones((self.num_edges, self.num_alpha))
        rr = np.expand_dims(r,axis = 1) * expansion
        RR_1 = np.expand_dims(self.R_1,axis = 1) * expansion
        RR_2 = np.expand_dims(self.R_2,axis = 1) * expansion

        z = (rr-RR_1**2)/(RR_2**2 - RR_1**2)
        z_mat = r_mat/(RR_2**2 - RR_1**2)

        fz = ( 6*(z)**5 - 15*(z)**4 + 10*(z)**3)
        fz = 3*z**2 - 2*z**3
        fz = (1 - np.cos(np.pi*z))/2

        fz_mat = (30*z**4 - 60*z**3 + 30*z**2) * z_mat
        fz_mat = (6*z - 6*z**2 ) * z_mat
        fz_mat = (np.sin(np.pi*z) * np.pi/2) * z_mat

        phi_alpha = np.where( (z <= 0) | (z >= 1), 0, fz_mat)

        return phi_alpha

    def damage_var_ntnn(self,r):
        np.seterr(divide='ignore')
        phi = np.zeros(r.shape)

        z = (r-self.R_1_ntnn**2)/(self.R_2_ntnn**2 - self.R_1_ntnn**2)

        fz = ( 6*(z)**5 - 15*(z)**4 + 10*(z)**3)
        fz = 3*z**2 - 2*z**3
        fz = (1 - np.cos(np.pi*z))/2

        phi = np.where(z <= 0, 0, fz)
        phi = np.where(z >= 1, 1, phi)

        return phi

    def damage_var_ntnn_mat(self,r,r_mat):
        np.seterr(divide='ignore',invalid='ignore')
        phi_alpha = np.zeros(r_mat.shape)
        expansion = np.ones((self.num_edges_ntnn, self.num_alpha))
        rr = np.expand_dims(r,axis = 1) * expansion
        RR_1 = np.expand_dims(self.R_1_ntnn,axis = 1) * expansion
        RR_2 = np.expand_dims(self.R_2_ntnn,axis = 1) * expansion

        z = (rr-RR_1**2)/(RR_2**2 - RR_1**2)
        z_mat = r_mat/(RR_2**2 - RR_1**2)

        fz = ( 6*(z)**5 - 15*(z)**4 + 10*(z)**3)
        fz = 3*z**2 - 2*z**3
        fz = (1 - np.cos(np.pi*z) )/2

        fz_mat = (30*z**4 - 60*z**3 + 30*z**2) * z_mat
        fz_mat = (6*z - 6*z**2 ) * z_mat
        fz_mat = (np.sin(np.pi*z)* np.pi/2) * z_mat

        phi_alpha = np.where( (z <= 0) | (z >= 1), 0, fz_mat)
        

        return phi_alpha


    def get_dirichlet_nodes(self):
        nodes = []
        for i,entry in enumerate(self.boundary):
##            print(entry)
            if entry == 'upper':
                cond_set_base = np.where(self.X[:,1] == self.pref_dist*np.sqrt(3)/2 * (self.L-1) )[0]
            elif entry == 'lower':
                cond_set_base = np.where(self.X[:,1] == 0)[0]  
            elif entry == 'upper_left':
                cond_set_base = np.where(self.X[:,1] == self.pref_dist*np.sqrt(3)/2 * (self.L-1) )[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,0] >= self.pref_dist*(self.B)*2/4 )[0])
            elif entry == 'lower_left':
                cond_set_base = np.where(self.X[:,1] == 0)[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,0] >= self.pref_dist*(self.B)*2/4 )[0])
            elif entry == 'upper_right':
                cond_set_base = np.where(self.X[:,1] == self.pref_dist*np.sqrt(3)/2 * (self.L-1) )[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,0] <= self.pref_dist*(self.B)*2/4)[0])
            elif entry == 'lower_right':
                cond_set_base = np.where(self.X[:,1] == 0)[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,0] <= self.pref_dist*(self.B)*2/4)[0])
            elif entry == 'left':
                cond_set_base = np.where(self.X[:,0] <= self.pref_dist/2)[0]
            elif entry == 'right':
                cond_set_base = np.where(self.X[:,0] >= self.pref_dist * (self.B-1))[0]
            elif entry == 'right_upper':
                cond_set_base = np.where(self.X[:,0] >= self.pref_dist * (self.B-1))[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,1] > 1/2 * self.pref_dist*np.sqrt(3)/2 * (self.L-1))[0])
            elif entry == 'right_lower':
                cond_set_base = np.where(self.X[:,0] >= self.pref_dist * (self.B-1))[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,1] <= 1/2 * self.pref_dist*np.sqrt(3)/2 * (self.L-1))[0])
            elif entry == 'left_upper':
                cond_set_base = np.where(self.X[:,0] <= self.pref_dist/2)[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,1] > 1/2 * self.pref_dist*np.sqrt(3)/2 * (self.L-1))[0])
            elif entry == 'left_lower':
                cond_set_base = np.where(self.X[:,0] <= self.pref_dist/2)[0]
                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,1] <= 1/2 * self.pref_dist*np.sqrt(3)/2 * (self.L-1))[0])
            else:
                print('unknown boundary type')

            if len(self.dim[i]) > 1:  
                cond_set = np.union1d(2*cond_set_base,2*cond_set_base+1)
            else:
                cond_set = 2*cond_set_base + self.dim[i][0]
            nodes.append(cond_set)
        return nodes

    def apply_material_parameters(self, edge_hist):

        self.alpha_inds= []
        self.alpha_fix_inds = []

        if self.chevron:        
            ind_c = np.where(edge_hist[:,:,0] - (self.B-1)*1/2 < (self.B-1)*1/4)[0]
            inda = np.where(abs(edge_hist[:,:,0] - 1/2 * 2/np.sqrt(3)*edge_hist[:,:,1]) + 10**-2>= (self.pref_dist)/2*(self.B-1) )[0]
            indb = np.where(abs(edge_hist[:,:,0] - 1/2 * 2/np.sqrt(3)*edge_hist[:,:,1]) - 10**-2<= (self.pref_dist)/2*(self.B-1) )[0]
            ind = np.intersect1d(inda,indb)
            ind = np.intersect1d(ind,ind_c)
            ind_2a = np.where(abs(edge_hist[:,:,0] - 1/2 * ((self.L-1) - 2/np.sqrt(3) *edge_hist[:,:,1]) ) + 10**-2>= (self.pref_dist)/2*(self.B-1))[0]
            ind_2b = np.where(abs(edge_hist[:,:,0] - 1/2 * ((self.L-1) - 2/np.sqrt(3) *edge_hist[:,:,1]) ) - 10**-2<= (self.pref_dist)/2*(self.B-1))[0]
            ind_2 = np.intersect1d(ind_2a,ind_2b)
            ind_2 = np.intersect1d(ind_2,ind_c)
            
            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds.append(ind)
            self.alpha_inds.append(ind_2)
        elif self.topdown:
            ind_a = np.where(edge_hist[:,:,0] > self.B * 1/3)[0]
            ind_b = np.where(edge_hist[:,:,1] > self.L  * np.sqrt(3)/4 )[0]
            ind = np.intersect1d(ind_a,ind_b)
            ind_2a = np.where(edge_hist[:,:,0] > self.B * 1/3)[0]
            ind_2b = np.where(edge_hist[:,:,1] < self.L  * np.sqrt(3)/4 - 5/10)[0]
            ind_2 = np.intersect1d(ind_2a,ind_2b)

            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds.append(ind)
            self.alpha_inds.append(ind_2)
        elif self.leftright:
            ind = np.where(edge_hist[:,:,0] > self.B * 1/2)[0]
            ind_2 = np.where(edge_hist[:,:,0] < self.B * 1/2)[0]


            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds.append(ind)
            self.alpha_inds.append(ind_2)
        elif self.circles:
            center1 = [(self.B-1) * 1/2, (self.L*np.sqrt(3)/2-1) * 1/2]
            center2 = [(self.B-1) * 3/4, (self.L*np.sqrt(3)/2-1) * 3/4]
            center3 = [(self.B-1) * 3/4, (self.L*np.sqrt(3)/2-1) * 1/4]
            ind = np.where((edge_hist[:,:,0]-center1[0])**2 +  (edge_hist[:,:,1]-center1[1])**2 <= (self.L/10)**2 +10**-1)[0]
            ind_2 = np.where((edge_hist[:,:,0]-center2[0])**2 +  (edge_hist[:,:,1]-center2[1])**2 <= (self.L/10)**2+10**-1)[0]
            ind_3 = np.where((edge_hist[:,:,0]-center3[0])**2 +  (edge_hist[:,:,1]-center3[1])**2 <= (self.L/10)**2+10**-1)[0]
            self.alpha_inds.append(ind_3)
            self.alpha_inds.append(ind_2)
        elif self.Friedrich_setup:
            #### all edges with both endpoints left of B/x
            ind_a = np.where(edge_hist[:,::2,0] < (self.B)*1.8/4 )[0]
            ind_b = np.where(edge_hist[:,1::2,0] < (self.B)*1.8/4 )[0]
            ind_left = np.intersect1d(ind_a,ind_b)
            
            #### all edges with both endpoints below L/2
            ind_below  = np.where(edge_hist[:,:,1] < (self.L-1)* np.sqrt(3)/2* 1/2)[0]
            
            #### all edges with both endpoints in a band around the center line with distance of at most 2 rows
            ind_band =  np.where(np.abs(edge_hist[:,:,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band_1 = np.where(np.abs(edge_hist[:,1::2,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band_2 = np.where(np.abs(edge_hist[:,::2,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band = np.intersect1d(ind_band_1, ind_band_2)
            
            #### edges located within each L1 circle
            ind_circle_main = np.where(np.abs(edge_hist[:,:,0] - self.B* 1.5/4)/1 + np.abs( edge_hist[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 5*(self.L+self.B)/82)[0]
            ind_circle_main = np.setdiff1d(range(self.num_edges), ind_circle_main)
            ind_circle_help = np.where(np.abs(edge_hist[:,:,0] - self.B* 1.6/4)/1 + np.abs( edge_hist[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 5*(self.L+self.B)/82)[0]
            ind_circle_help = np.setdiff1d(range(self.num_edges), ind_circle_help)
            ind_circle_up = np.where(np.abs(edge_hist[:,:,0] - self.B* 3/4)/1 + np.abs( edge_hist[:,:,1] - 3/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 4*(self.L+self.B)/82)[0]
            ind_circle_up = np.setdiff1d(range(self.num_edges), ind_circle_up)
            ind_circle_low_1 = np.where(np.abs(edge_hist[:,:,0] - self.B* 2.5/4)/1 + np.abs( edge_hist[:,:,1] - 1/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 3*(self.L+self.B)/82)[0]
            ind_circle_low_1 = np.setdiff1d(range(self.num_edges), ind_circle_low_1)
            ind_circle_low_2= np.where(np.abs(edge_hist[:,:,0] - self.B* 3.5/4)/1 + np.abs( edge_hist[:,:,1] - 1/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 3*(self.L+self.B)/82)[0]
            ind_circle_low_2 = np.setdiff1d(range(self.num_edges), ind_circle_low_2)
            ind_circle_low = np.union1d(ind_circle_low_2, ind_circle_low_1)

            #### circle at the left 
            ind_circle_edges = ind_circle_main #
            ind_circle_edges = np.setdiff1d(ind_circle_main, ind_circle_help) ### out and inner boundaries divided
            ind_circle_rest = np.setdiff1d(ind_circle_main, ind_circle_edges) ### edges without the outer boundary

            #### path to circle at left
            ind_left_band = np.intersect1d(ind_band, ind_left)
            ind_left_band = np.setdiff1d(ind_left_band, ind_circle_rest)

            #### upper and lower paths around the left cirlce           
            ind_upper_main = np.intersect1d(ind_circle_edges, ind_below)
            ind_lower_main = np.setdiff1d(ind_circle_edges, ind_upper_main)
            ind_upper_main = np.setdiff1d(ind_upper_main, ind_left_band)
            ind_lower_main = np.setdiff1d(ind_lower_main, ind_left_band)

            
            ind = ind_upper_main
            ind_2 = ind_lower_main
            ind_3 = np.union1d(ind_left_band, np.union1d(ind_circle_low, ind_circle_up))
            ind_3 = np.union1d(ind_3, ind_circle_rest)
            
            self.alpha_inds.append(ind)
            self.alpha_inds.append(ind_2)
            self.alpha_fix_inds.append(ind_left_band)
            self.alpha_fix_inds.append(ind_circle_rest)
            self.alpha_fix_inds.append(ind_circle_low)
            self.alpha_fix_inds.append(ind_circle_up)
        elif self.Friedrich_setup_2:
            #### all edges with both endpoints left of B/x
            ind_a = np.where(edge_hist[:,::2,0] < (self.B)*1.8/4 )[0]
            ind_b = np.where(edge_hist[:,1::2,0] < (self.B)*1.8/4 )[0]
            ind_left = np.intersect1d(ind_a,ind_b)
            
            #### all edges with both endpoints below L/2
            ind_below  = np.where(edge_hist[:,:,1] < (self.L-1)* np.sqrt(3)/2* 1/2)[0]
            
            #### all edges with both endpoints in a band around the center line with distance of at most 2 rows
            ind_band =  np.where(np.abs(edge_hist[:,:,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band_1 = np.where(np.abs(edge_hist[:,1::2,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band_2 = np.where(np.abs(edge_hist[:,::2,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]
            ind_band = np.intersect1d(ind_band_1, ind_band_2)
            
            #### edges located within each L1 circle
            ind_circle_main = np.where(np.abs(edge_hist[:,:,0] - self.B* 1.5/4)/1 + np.abs( edge_hist[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 5*(self.L+self.B)/82)[0]
            ind_circle_main = np.setdiff1d(range(self.num_edges), ind_circle_main)
            ind_circle_help = np.where(np.abs(edge_hist[:,:,0] - self.B* 1.6/4)/1 + np.abs( edge_hist[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) >= 5*(self.L+self.B)/82)[0]
            ind_circle_help = np.setdiff1d(range(self.num_edges), ind_circle_help)
            scale = 1.5
            ind_circle_up = np.where(np.abs(edge_hist[:,:,0] - self.B* 2.9/4)/scale + np.abs( edge_hist[:,:,1] - 4/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) >= 4*(self.L+self.B)/82)[0]

            ind_circle_up = np.setdiff1d(range(self.num_edges), ind_circle_up)

            
            #### change this back to 2.5, 3.5
            ind_circle_low_1 = np.where(np.abs(edge_hist[:,:,0] - self.B* 2.4/4)/scale + np.abs( edge_hist[:,:,1] - 0/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) >= 3*(self.L+self.B)/82)[0]
            ind_circle_low_1 = np.setdiff1d(range(self.num_edges), ind_circle_low_1)
            ind_circle_low_2= np.where(np.abs(edge_hist[:,:,0] - self.B* 3.4/4)/scale + np.abs( edge_hist[:,:,1] - 0/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) >= 3*(self.L+self.B)/82)[0]
            ind_circle_low_2 = np.setdiff1d(range(self.num_edges), ind_circle_low_2)
            ind_circle_low = np.union1d(ind_circle_low_2, ind_circle_low_1)

            #### circle at the left 
            ind_circle_edges = ind_circle_main #
            ind_circle_edges = np.setdiff1d(ind_circle_main, ind_circle_help) ### out and inner boundaries divided
            ind_circle_rest = np.setdiff1d(ind_circle_main, ind_circle_edges) ### edges without the outer boundary

            #### path to circle at left
            ind_left_band = np.intersect1d(ind_band, ind_left)
            ind_left_band = np.setdiff1d(ind_left_band, ind_circle_rest)

            #### upper and lower paths around the left cirlce           
            ind_upper_main = np.intersect1d(ind_circle_edges, ind_below)
            ind_lower_main = np.setdiff1d(ind_circle_edges, ind_upper_main)
            ind_upper_main = np.setdiff1d(ind_upper_main, ind_left_band)
            ind_lower_main = np.setdiff1d(ind_lower_main, ind_left_band)

            
            ind = ind_upper_main
            ind_2 = ind_lower_main
            ind_3 = np.union1d(ind_left_band, np.union1d(ind_circle_low, ind_circle_up))
            ind_3 = np.union1d(ind_3, ind_circle_rest)
            
            self.alpha_inds.append(ind)
            self.alpha_inds.append(ind_2)
            self.alpha_fix_inds.append(ind_left_band)
            self.alpha_fix_inds.append(ind_circle_rest)
            self.alpha_fix_inds.append(ind_circle_low)
            self.alpha_fix_inds.append(ind_circle_up)
            
            
        self.pots = np.ones((self.num_edges,))*-1
        self.pots_og = self.pots.copy()
        self.alpha_vec = np.ones((self.num_edges,))
        if self.Friedrich_setup or self.Friedrich_setup_2: 
            for k,entry in enumerate(self.alpha_fix_inds):
                self.alpha_vec[entry] = self.alpha_fix_list[k]
        for k,entry in enumerate(self.alpha_inds):
            self.alpha_vec[entry] = self.alpha_list[k]

        self.pots = self.alpha_vec* self.pots
        
        return
    
    def apply_material_parameters_ntnn(self, edge_hist_ntnn):
        
        self.alpha_inds_ntnn= []
        self.alpha_fix_inds_ntnn = []

        if self.chevron:            
            ind_c = np.where(edge_hist_ntnn[:,:,0] - (self.B-1)*1/2 < (self.B-1)*1/4)[0]
            inda = np.where(abs(edge_hist_ntnn[:,:,0] - 1/2 * 2/np.sqrt(3)*edge_hist_ntnn[:,:,1]) + 10**-2>= (self.pref_dist)/2*(self.B-1) )[0]
            indb = np.where(abs(edge_hist_ntnn[:,:,0] - 1/2 * 2/np.sqrt(3)*edge_hist_ntnn[:,:,1]) - 10**-2<= (self.pref_dist)/2*(self.B-1) )[0]
            ind = np.intersect1d(inda,indb)
            ind = np.intersect1d(ind,ind_c)    
            ind_2a = np.where(abs(edge_hist_ntnn[:,:,0] - 1/2 * ((self.L-1) - 2/np.sqrt(3) *edge_hist_ntnn[:,:,1]) ) + 10**-2>= (self.pref_dist)/2*(self.B-1))[0]
            ind_2b = np.where(abs(edge_hist_ntnn[:,:,0] - 1/2 * ((self.L-1) - 2/np.sqrt(3) *edge_hist_ntnn[:,:,1]) ) - 10**-2<= (self.pref_dist)/2*(self.B-1))[0]
            ind_2 = np.intersect1d(ind_2a,ind_2b)
            ind_2 = np.intersect1d(ind_2,ind_c)
            
            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds_ntnn.append(ind)
            self.alpha_inds_ntnn.append(ind_2)
        elif self.topdown:
            ind_a = np.where(edge_hist_ntnn[:,:,0] > self.B * 1/3)[0]
            ind_b = np.where(edge_hist_ntnn[:,:,1] > self.L  * np.sqrt(3)/4 )[0]
            ind = np.intersect1d(ind_a,ind_b)
            ind_2a = np.where(edge_hist_ntnn[:,:,0] > self.B * 1/3)[0]
            ind_2b = np.where(edge_hist_ntnn[:,:,1] < self.L  * np.sqrt(3)/4 - 5/10)[0]
            ind_2 = np.intersect1d(ind_2a,ind_2b)

            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds_ntnn.append(ind)
            self.alpha_inds_ntnn.append(ind_2)
        elif self.leftright:
            ind = np.where(edge_hist_ntnn[:,:,0] > self.B * 1/3)[0]
            ind_2 = np.where(edge_hist_ntnn[:,:,0] < self.B * 1/3)[0]

            ind_3 = np.intersect1d(ind_2,ind)
            ind = np.setdiff1d(ind,ind_3)
            ind_2 = np.setdiff1d(ind_2,ind_3)
            self.alpha_inds_ntnn.append(ind)
            self.alpha_inds_ntnn.append(ind_2)

        elif self.circles:
            center1 = [(self.B-1) * 1/2, (self.L*np.sqrt(3)/2-1) * 1/2]
            center2 = [(self.B-1) * 3/4, (self.L*np.sqrt(3)/2-1) * 3/4]
            center3 = [(self.B-1) * 3/4, (self.L*np.sqrt(3)/2-1) * 1/4]
            ind = np.where((edge_hist_ntnn[:,:,0]-center1[0])**2 +  (edge_hist_ntnn[:,:,1]-center1[1])**2 <= (self.L/10)**2 +10**-1)[0]
            ind_2 = np.where((edge_hist_ntnn[:,:,0]-center2[0])**2 +  (edge_hist_ntnn[:,:,1]-center2[1])**2 <= (self.L/10)**2+10**-1)[0]
            ind_3 = np.where((edge_hist_ntnn[:,:,0]-center3[0])**2 +  (edge_hist_ntnn[:,:,1]-center3[1])**2 <= (self.L/10)**2+10**-1)[0]
            self.alpha_inds_ntnn.append(ind_3)
            self.alpha_inds_ntnn.append(ind_2)
        elif self.Friedrich_setup:
            #### all edges with both endpoints left of B/x
            ind_a = np.where(edge_hist_ntnn[:,::2,0] < (self.B)*1.8/4 )[0]
            ind_b = np.where(edge_hist_ntnn[:,1::2,0] < (self.B)*1.8/4 )[0]
            ind_left = np.intersect1d(ind_a,ind_b)
            #### all edges with both endpoints below L/2

            ind_below  = np.where(edge_hist_ntnn[:,:,1] < (self.L-1)* np.sqrt(3)/2* 1/2)[0]
            
            #### all edges with both endpoints in a band around the center line with distance of at most 2 rows
            ind_band =  np.where(np.abs(edge_hist_ntnn[:,:,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]

            #### edges located within each L1 circle
            ind_circle_main = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_1 = np.where(np.abs(edge_hist_ntnn[:,1::2,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,1::2,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_2 = np.where(np.abs(edge_hist_ntnn[:,::2,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,::2,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_inner = np.setdiff1d(ind_circle_main_1, ind_circle_main_2)
            ind_circle_help = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 1.67/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]

            ind_circle_up = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 3/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 3/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 4)[0]
            ind_circle_low_1 = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 2.5/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 3)[0]
            ind_circle_low_2= np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 3.5/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/4 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 3)[0]
            ind_circle_low = np.union1d(ind_circle_low_2, ind_circle_low_1)


            ind_circle_edges = np.setdiff1d(ind_circle_main, ind_circle_help)
            ind_circle_rest = np.setdiff1d(ind_circle_main, ind_circle_edges)

            ind_left_band = np.intersect1d(ind_band, ind_left)
            ind_left_band = np.setdiff1d(ind_left_band, ind_circle_rest)

            ind_upper_main = np.intersect1d(ind_circle_edges, ind_below)
            ind_lower_main = np.setdiff1d(ind_circle_edges, ind_upper_main)

            ind_upper_main = np.setdiff1d(ind_upper_main, ind_left_band)
            ind_lower_main = np.setdiff1d(ind_lower_main, ind_left_band)
            
            ind = ind_upper_main
            ind_2 = ind_lower_main
            ind_3 = np.union1d(ind_left_band, np.union1d(ind_circle_low, ind_circle_up))
            ind_3 = np.union1d(ind_3, ind_circle_rest)

            self.alpha_inds_ntnn.append(ind)
            self.alpha_inds_ntnn.append(ind_2)
            self.alpha_fix_inds_ntnn.append(ind_left_band)
            self.alpha_fix_inds_ntnn.append(ind_circle_rest)
            self.alpha_fix_inds_ntnn.append(ind_circle_low)
            self.alpha_fix_inds_ntnn.append(ind_circle_up)

        elif self.Friedrich_setup_2:
            #### all edges with both endpoints left of B/x
            ind_a = np.where(edge_hist_ntnn[:,::2,0] < (self.B)*1.8/4 )[0]
            ind_b = np.where(edge_hist_ntnn[:,1::2,0] < (self.B)*1.8/4 )[0]
            ind_left = np.intersect1d(ind_a,ind_b)
            #### all edges with both endpoints below L/2

            ind_below  = np.where(edge_hist_ntnn[:,:,1] < (self.L-1)* np.sqrt(3)/2* 1/2)[0]
            
            #### all edges with both endpoints in a band around the center line with distance of at most 2 rows
            ind_band =  np.where(np.abs(edge_hist_ntnn[:,:,1]-(self.L-1)* np.sqrt(3)/2* 1/2 ) <  1/np.sqrt(3) )[0]

            #### edges located within each L1 circle
            ind_circle_main = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_1 = np.where(np.abs(edge_hist_ntnn[:,1::2,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,1::2,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_2 = np.where(np.abs(edge_hist_ntnn[:,::2,0] - self.B* 1.5/4)/1 + np.abs( edge_hist_ntnn[:,::2,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]
            ind_circle_main_inner = np.setdiff1d(ind_circle_main_1, ind_circle_main_2)
            ind_circle_help = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 1.67/4)/1 + np.abs( edge_hist_ntnn[:,:,1] - 1/2 * (self.L-1) * np.sqrt(3)/2)/(np.sqrt(3)) < 5)[0]

            scale = 1.5
            ind_circle_up = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 2.9/4)/scale + np.abs( edge_hist_ntnn[:,:,1] - 4/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) < 4)[0]
            
            #### change this back to 2.5, 3.5
            ind_circle_low_1 = np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 2.4/4)/scale + np.abs( edge_hist_ntnn[:,:,1] - 0/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) < 3)[0]
            ind_circle_low_2= np.where(np.abs(edge_hist_ntnn[:,:,0] - self.B* 3.4/4)/scale + np.abs( edge_hist_ntnn[:,:,1] - 0/4 * (self.L-1) * np.sqrt(3)/2)/(scale*np.sqrt(3)) < 3)[0]
            ind_circle_low = np.union1d(ind_circle_low_2, ind_circle_low_1)

            ind_circle_edges = np.setdiff1d(ind_circle_main, ind_circle_help)
            ind_circle_rest = np.setdiff1d(ind_circle_main, ind_circle_edges)

            ind_left_band = np.intersect1d(ind_band, ind_left)
            ind_left_band = np.setdiff1d(ind_left_band, ind_circle_rest)

            ind_upper_main = np.intersect1d(ind_circle_edges, ind_below)
            ind_lower_main = np.setdiff1d(ind_circle_edges, ind_upper_main)

            ind_upper_main = np.setdiff1d(ind_upper_main, ind_left_band)
            ind_lower_main = np.setdiff1d(ind_lower_main, ind_left_band)
            
            ind = ind_upper_main
            ind_2 = ind_lower_main
            ind_3 = np.union1d(ind_left_band, np.union1d(ind_circle_low, ind_circle_up))
            ind_3 = np.union1d(ind_3, ind_circle_rest)

            self.alpha_inds_ntnn.append(ind)
            self.alpha_inds_ntnn.append(ind_2)
            self.alpha_fix_inds_ntnn.append(ind_left_band)
            self.alpha_fix_inds_ntnn.append(ind_circle_rest)
            self.alpha_fix_inds_ntnn.append(ind_circle_low)
            self.alpha_fix_inds_ntnn.append(ind_circle_up)

        self.pots_ntnn = np.ones((self.num_edges_ntnn,))*-1
        self.pots_ntnn_og = self.pots_ntnn.copy()
        self.alpha_vec_ntnn = np.ones((self.num_edges_ntnn,))

        if self.Friedrich_setup or self.Friedrich_setup_2: 
            for k,entry in enumerate(self.alpha_fix_inds_ntnn):
                self.alpha_vec_ntnn[entry] = self.alpha_fix_list[k]

        for k,entry in enumerate(self.alpha_inds_ntnn):
            self.alpha_vec_ntnn[entry] = self.alpha_list[k]
                
        self.pots_ntnn =  self.alpha_vec_ntnn * self.pots_ntnn
        
        return 

    def Lennard_jones(self,r,pot_min = -1):
        q = self.pref_dist**2/(np.cbrt(2)*r)
        res = -4*pot_min*(q**6-q**3)
        return res
    
    def Lennard_jones_der(self,r,pot_min = -1):
        res = 6*pot_min*((self.pref_dist**12)/(r**7) - (self.pref_dist**6)/(r**4))
        return res
    
    def Lennard_jones_hess(self,r,pot_min = -1):
        res = 6*pot_min*(-(7*self.pref_dist**12)/(r**8) + (4*self.pref_dist**6)/(r**5) )
        return res

    def energy(self, y, phi, dist_hist, y_full):
        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()
        dist = np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2
        pots = self.Lennard_jones(dist,self.pots)
        pots_max = self.Lennard_jones(dist_hist,self.pots)
        maxdist = np.maximum(dist,dist_hist)
        
        pots_max = self.softmax(pots,pots_max)
        eny = 1/2*(np.inner((1-phi),pots) + np.inner(phi,pots_max))
        return eny

    def energy_mat(self, y, phi, dist_hist, y_full, y_alpha, phi_alpha, dist_hist_alpha):
        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()

        z_alpha = np.zeros((self.L*self.B*2,self.num_alpha))
        z_alpha[self.free_nodes,:] = y_alpha
        y_alpha = z_alpha.copy()
        
        dist = np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2

        dist_alpha = 2* ( np.expand_dims(np.inner(self.edge_index.T, y[::2]), axis = 1) * np.inner(self.edge_index.T, y_alpha[::2,:].T)
                            + np.expand_dims(np.inner(self.edge_index.T, y[1::2]), axis = 1) * np.inner(self.edge_index.T, y_alpha[1::2,:].T ) )

        pots = np.expand_dims(self.Lennard_jones(dist,self.pots), axis = 1)        
        pots_1d = self.Lennard_jones(dist,self.pots)
        pots_max_f = self.Lennard_jones(dist_hist,self.pots)
        maxdist = np.maximum(dist,dist_hist)
        pots_max = self.softmax(pots_1d,pots_max_f)

        pots_der = self.Lennard_jones_der(dist,self.pots)
        pots_der_max = self.Lennard_jones_der(dist_hist,self.pots)

        pots_1d_mat = self.Lennard_jones(dist,self.pots_og)
        pots_max_f_mat = self.Lennard_jones(dist_hist,self.pots_og)
        pots_alpha = np.zeros((self.num_edges,self.num_alpha))
        pots_max_f_alpha = np.zeros((self.num_edges,self.num_alpha))

        count = 0
        for entry in self.alpha_inds:
            
            pots_alpha[entry,count] = pots_1d_mat[entry]
            pots_max_f_alpha[entry,count] = pots_max_f_mat[entry]
            count +=1

        eny_alpha = np.zeros(self.num_alpha)
        for ii in range(self.num_alpha):

            eny_alpha[ii] = 1/2*(   np.inner((1-phi),pots_der*dist_alpha[:,ii]) 
                                    + np.inner(phi, self.softmax_der(pots_1d, pots_max_f, pots_der*dist_alpha[:,ii], pots_der_max * dist_hist_alpha[:,ii]) )
                                    + np.inner((1-phi),pots_alpha[:,ii]) + np.inner(phi,self.softmax_der(pots_1d, pots_max_f, pots_alpha[:,ii], pots_max_f_alpha[:,ii]) )
                                    + np.inner((-phi_alpha[:,ii]),pots_1d) + np.inner(phi_alpha[:,ii],pots_max)
                                 )
        return eny_alpha
        

    def get_triangles(self,x,Return_Flag = False):
        y = np.reshape(x,(self.L,self.B,2))
        triang = np.zeros(((self.B-1)*(self.L-1)*2,3,2))
        if Return_Flag: 
            triang_list = np.zeros(((self.B-1)*(self.L-1)*2,3,))
            self.triang_list = np.zeros(((self.B-1)*(self.L-1)*2,3,self.B*self.L))
            indeces = np.reshape(np.arange(0,self.B*self.L), (self.L,self.B))
        tid = 0
        
        ## up even rows
        for j in range(self.B-1):
            for i in range(0,self.L-1,2):
                triang[tid,0,:] = y[i,j]
                triang[tid,1,:] = y[i,j+1]
                triang[tid,2,:] = y[i+1,j]
                if Return_Flag:
                    triang_list[tid,0] = indeces[i,j]
                    triang_list[tid,1] = indeces[i,j+1]
                    triang_list[tid,2] = indeces[i+1,j]
                    self.triang_list[tid,0,indeces[i,j]] = 1
                    self.triang_list[tid,1,indeces[i,j+1]] = 1
                    self.triang_list[tid,2,indeces[i+1,j]] = 1
                tid +=1

        ## up odd rows
        for j in range(self.B-1):
            for i in range(1,self.L-1,2):
                triang[tid,0,:] = y[i,j]
                triang[tid,1,:] = y[i,j+1]
                triang[tid,2,:] = y[i+1,j+1]
                if Return_Flag:
                    triang_list[tid,0] = indeces[i,j]
                    triang_list[tid,1] = indeces[i,j+1]
                    triang_list[tid,2] = indeces[i+1,j+1]
                    self.triang_list[tid,0,indeces[i,j]] = 1
                    self.triang_list[tid,1,indeces[i,j+1]] = 1
                    self.triang_list[tid,2,indeces[i+1,j+1]] = 1
                tid +=1
                
        ## down even rows
        for j in range(self.B-1):
            for i in range(2,self.L,2):
                triang[tid,0,:] = y[i,j]
                triang[tid,1,:] = y[i-1,j]
                triang[tid,2,:] = y[i,j+1]
                if Return_Flag:
                    triang_list[tid,0] = indeces[i,j]
                    triang_list[tid,1] = indeces[i-1,j]
                    triang_list[tid,2] = indeces[i,j+1]
                    self.triang_list[tid,0,indeces[i,j]] = 1
                    self.triang_list[tid,1,indeces[i-1,j]] = 1
                    self.triang_list[tid,2,indeces[i,j+1]] = 1
                tid +=1
                
        ## down odd rows
        for j in range(self.B-1):
            for i in range(1,self.L,2):
                triang[tid,0,:] = y[i,j]
                triang[tid,1,:] = y[i-1,j+1]
                triang[tid,2,:] = y[i,j+1]
                if Return_Flag:
                    triang_list[tid,0] = indeces[i,j]
                    triang_list[tid,1] = indeces[i-1,j+1]
                    triang_list[tid,2] = indeces[i,j+1]
                    self.triang_list[tid,0,indeces[i,j]] = 1
                    self.triang_list[tid,1,indeces[i-1,j+1]] = 1
                    self.triang_list[tid,2,indeces[i,j+1]] = 1
                tid +=1
                
        if Return_Flag:
            return triang, triang_list.astype(int)
        else: 
            return triang
            

    def get_gradients_new(self,y, Return_Flag = False):
        if Return_Flag:
            triang,triang_list = self.get_triangles(y,Return_Flag)
        else:
            ### assume get_triangles was run once before
            triang = np.zeros(((self.B-1)*(self.L-1)*2,3,2))
            triang[:,:,0] = np.squeeze(np.inner(self.triang_list, y[::2].T))
            triang[:,:,1] = np.squeeze(np.inner(self.triang_list, y[1::2].T))
            
        grad11 = triang[:,2,0] - triang[:,1,0]
        grad21 = triang[:,2,1] - triang[:,1,1]
        p = 1/np.sqrt(3)
        grad22 = p * ( 2*(triang[:,2,1] - triang[:,0,1]) - (triang[:,1,1]- triang[:,0,1]))
        grad12 = p * ( 2*(triang[:,2,0] - triang[:,0,0]) - (triang[:,1,0]- triang[:,0,0]))
        if Return_Flag:
            gradlist1 = np.zeros((grad11.shape[0],self.B*self.L))
            gradlist2 = np.zeros((grad11.shape[0],self.B*self.L))
            for k in range(grad11.shape[0]):
                gradlist1[k,triang_list[k,2]] = 1
                gradlist1[k,triang_list[k,1]] = -1
                gradlist2[k,triang_list[k,2]] = 2*p
                gradlist2[k,triang_list[k,1]] = -p
                gradlist2[k,triang_list[k,0]] = -p
                
        grads = np.concatenate((grad11,grad12,grad21,grad22))
        if Return_Flag:
            gradlist = np.concatenate((gradlist1,gradlist2))
            self.gradlist = gradlist

            return grads, gradlist
        else: 
            return grads

    def L2_dissipation(self,y,y_1,y_full):
        x = np.zeros((self.L*self.B*2,))
        x[self.dir_nodes] = y_full[self.dir_nodes]
        x[self.free_nodes] = y
        diss = 1/2 * self.nu/(self.T[-1]/self.n_time_steps_og) * np.inner(x-y_1,x-y_1)
        return diss

    def L2_dissipation_mat(self,y,y_1,y_full, y_alpha, y_1_alpha):
        x = np.zeros((self.L*self.B*2,))
        x_alpha = np.zeros((self.L*self.B*2,self.num_alpha))
        x[self.dir_nodes] = y_full[self.dir_nodes]
        x[self.free_nodes] = y
        x_alpha[self.free_nodes] = y_alpha
        diss_alpha = self.nu/(self.T[-1]/self.n_time_steps_og) * ((x-y_1)@(x_alpha - y_1_alpha))
        return diss_alpha

    def KV_dissipation(self,y,y_1, y_full):
        x = np.zeros((self.L*self.B*2,))
        x[self.dir_nodes] = y_full[self.dir_nodes]
        x[self.free_nodes] = y
        grad_app = self.get_gradients_new(x-y_1)
        diss = 1/2 * self.nu/(self.T[-1]/self.n_time_steps_og) * ( np.inner(grad_app , grad_app ))
        return diss
    
    def energy_ntnn(self, y, phi, dist_hist, phi_ntnn, dist_hist_ntnn, y_full):

        eny = self.energy(y,phi,dist_hist,y_full)

        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()
        
        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)
        pots_ntnn = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_ntnn = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        maxdist_ntnn = np.maximum(dist_ntnn,dist_hist_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn,pots_max_ntnn)
        
        eny += self.ntnn_weight * 1/2 * (np.inner((1-phi_ntnn),pots_ntnn) + np.inner(phi_ntnn, pots_max_ntnn))
        return eny

    def energy_ntnn_mat(self, y, phi, dist_hist, phi_ntnn, dist_hist_ntnn, y_full, y_alpha, phi_alpha, dist_hist_alpha , phi_ntnn_alpha, dist_hist_ntnn_alpha):
        eny_alpha = self.energy_mat( y, phi, dist_hist, y_full, y_alpha, phi_alpha, dist_hist_alpha)

        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()
        
        z_alpha = np.zeros((self.L*self.B*2,self.num_alpha))
        z_alpha[self.free_nodes,:] = y_alpha
        y_alpha = z_alpha.copy()

        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)


        dist_ntnn_alpha = 2/3* ( np.expand_dims(np.inner(self.edge_index_ntnn.T, y[::2]), axis = 1) * np.inner(self.edge_index_ntnn.T, y_alpha[::2,:].T)
                            + np.expand_dims(np.inner(self.edge_index_ntnn.T, y[1::2]), axis = 1) * np.inner(self.edge_index_ntnn.T, y_alpha[1::2,:].T ) )

        
        pots_ntnn = np.expand_dims(self.Lennard_jones(dist_ntnn,self.pots_ntnn), axis = 1)
        pots_ntnn_1d = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_f_ntnn = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        maxdist_ntnn = np.maximum(dist_ntnn,dist_hist_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn_1d,pots_max_f_ntnn)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)
        pots_der_ntnn_max = self.Lennard_jones_der(dist_hist_ntnn,self.pots_ntnn)

        pots_ntnn_1d_mat = self.Lennard_jones(dist_ntnn,self.pots_ntnn_og)
        pots_max_f_ntnn_mat = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn_og)

        pots_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))
        pots_max_f_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))

        count = 0
        for entry in self.alpha_inds_ntnn:
            

            pots_ntnn_alpha[entry,count] = pots_ntnn_1d_mat[entry]
            pots_max_f_ntnn_alpha[entry,count] = pots_max_f_ntnn_mat[entry]
            count +=1
            
        for ii in range(self.num_alpha):
            eny_alpha[ii] += self.ntnn_weight * 1/2 * (
                                np.inner((1-phi_ntnn), pots_der_ntnn * dist_ntnn_alpha[:,ii]) + np.inner(phi_ntnn, self.softmax_der(pots_ntnn_1d, pots_max_f_ntnn, pots_der_ntnn * dist_ntnn_alpha[:,ii], pots_der_ntnn_max * dist_hist_ntnn_alpha[:,ii]) )
                                + np.inner((1-phi_ntnn),pots_ntnn_alpha[:,ii]) + np.inner(phi_ntnn, self.softmax_der(pots_ntnn_1d, pots_max_f_ntnn, pots_ntnn_alpha[:,ii], pots_max_f_ntnn_alpha[:,ii]))
                                + np.inner((-phi_ntnn_alpha[:,ii]),pots_ntnn_1d) + np.inner(phi_ntnn_alpha[:,ii], pots_max_ntnn)
                                )
        return eny_alpha

    def smoothmax(self,a,b):
        res = 1/2 * ( a+b + np.sqrt((a-b)**2 + self.smoothmaxalpha) )
        return res
    
    def smoothmax_der(self, a,b,dera, derb):
        res = 1/2 * ( dera + derb +  (a-b) * (dera-derb)/np.sqrt((a-b)**2 + self.smoothmaxalpha) )
        return res

    def softexp(self,a):
        res = exp(a)
        res = np.where(res > 10**12, 10**12, res)
        return res

    def softmax(self,out1,out2):

        np.seterr(over='ignore')
        ret = ( out1 * self.softexp(self.softmaxalpha * out1) + out2 * self.softexp(self.softmaxalpha * out2) )/ ( self.softexp(self.softmaxalpha * out1) + self.softexp(self.softmaxalpha * out2) )
        return ret


    def softmax_der(self,out1,out2, in1, in2):
        np.seterr(over='ignore')
        ret = ( ( ( self.softexp(2 * self.softmaxalpha * out1) + (1 + self.softmaxalpha * (out1 - out2)) * self.softexp(self.softmaxalpha * (out1 + out2)) ) * in1 +
                  ( self.softexp(2 * self.softmaxalpha * out2) + (1 + self.softmaxalpha * (out2 - out1)) * self.softexp(self.softmaxalpha * (out1 + out2)) ) * in2 ) /
                ( self.softexp(self.softmaxalpha * out1) + self.softexp(self.softmaxalpha * out2) )**2 )
        return ret
    def softmax_hess(self, out1, out2, in1, in2, outder1, outder2, inder1, inder2 ):
        np.seterr(over='ignore')

        divisor = (self.softexp(self.softmaxalpha * out1) + self.softexp(self.softmaxalpha * out2) )
        pp_a = ( self.softexp(2 * self.softmaxalpha * out1) + (1 + self.softmaxalpha * (out1 - out2)) * self.softexp(self.softmaxalpha * (out1 + out2)) ) 
        pp_b = ( self.softexp(2 * self.softmaxalpha * out2) + (1 + self.softmaxalpha * (out2 - out1)) * self.softexp(self.softmaxalpha * (out1 + out2)) ) 

        part2_a = ( 2*self.softmaxalpha * outder1 * self.softexp(2 * self.softmaxalpha * out1)
                  + ( (1 + self.softmaxalpha * (out1 - out2)) * self.softmaxalpha * (outder1 + outder2) + self.softmaxalpha * (outder1 - outder2)  )
                  * self.softexp(self.softmaxalpha * (out1 + out2) ) ) * in1
        part2_b = ( 2*self.softmaxalpha * outder2 * self.softexp(2 * self.softmaxalpha * out2)
                  + ( (1 + self.softmaxalpha * (out2 - out1)) * self.softmaxalpha * (outder1 + outder2) + self.softmaxalpha * (outder2 - outder1)  )
                  * self.softexp(self.softmaxalpha * (out1 + out2) ) ) * in2

        divisor_deriv = self.softmaxalpha * outder1 * self.softexp(self.softmaxalpha * out1 ) + self.softmaxalpha * outder2 * self.softexp(self.softmaxalpha * out2 )

        ret = (part2_a + part2_b)/divisor**2 + (pp_a * inder1 + pp_b * inder2)/divisor**2 - 2 * (pp_a* in1 + pp_b* in2) * divisor_deriv/divisor**3
        
        return ret

    def Jacobian(self,y,phi,dist_hist,y_1, hist_edges, edge_index,y_full, reshape = True):
        J = np.zeros((self.L*self.B*2,))
        RJ = np.zeros((self.L*self.B*2,))
        if reshape: 
            z = np.zeros((self.L*self.B*2,))
            z[self.dir_nodes] = y_full[self.dir_nodes]
            z[self.free_nodes] = y
            y = z.copy()
        
        n = self.L*self.B*2
        x = np.reshape(y,(self.L*self.B,2))
        x_1 = np.reshape(y_1,(self.L*self.B,2))

        dist =  (np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2)
        maxdist = np.maximum(dist,dist_hist)

        pots = self.Lennard_jones(dist,self.pots)
        pots_max_f = self.Lennard_jones(dist_hist,self.pots)
        pots_max = self.softmax(pots,pots_max_f)

        pots_der = self.Lennard_jones_der(dist,self.pots)

        el_1 = np.inner(self.edge_index.T, y[::2])
        el_2 = np.inner(self.edge_index.T, y[1::2])
        
        if self.diss_type == 'KV':
            grad_app = self.get_gradients_new(y-y_1)
            grads_list = self.gradlist

        if self.nu != 0:     
            if self.diss_type == 'L2':                
                RJ[::2] = self.nu/(self.T[-1]/self.n_time_steps_og) * (y[::2] - y_1[::2])
                RJ[1::2] = self.nu/(self.T[-1]/self.n_time_steps_og) * (y[1::2] - y_1[1::2])
            elif self.diss_type == 'KV':
                RJ[::2] = self.nu/(self.T[-1]/self.n_time_steps_og) * np.inner(self.gradlist.T, grad_app[:self.gradlist.shape[0]].T)
                RJ[1::2] = self.nu/(self.T[-1]/self.n_time_steps_og) * np.inner(self.gradlist.T, grad_app[self.gradlist.shape[0]:].T)
            else:
                print('assign dissipation type')
                
        J[::2] = ( np.inner(self.edge_index,(1-phi).T * pots_der.T * el_1.T  )
                       +np.inner(self.edge_index, phi.T * self.softmax_der(pots, pots_max_f, pots_der * el_1, 0).T) ) 
        J[1::2] = ( np.inner( self.edge_index, (1-phi).T * pots_der.T *  el_2.T)
                     +np.inner(self.edge_index ,phi.T * self.softmax_der(pots, pots_max_f, pots_der * el_2, 0).T) )
        J += RJ
        return J[self.free_nodes]
    
    def Jacobian_ntnn(self,y,phi,dist_hist,y_1, hist_edges, edge_index,phi_ntnn,dist_hist_ntnn, hist_edges_ntnn, edge_index_ntnn, y_full):
        J_ntnn = np.zeros((self.L*self.B*2,))

        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()
        
        J = self.Jacobian(y,phi,dist_hist,y_1, hist_edges, edge_index,y_full, reshape = False)
        
        x = np.reshape(y,(self.L*self.B,2))
        
        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)
        maxdist = np.maximum(dist_ntnn,dist_hist_ntnn)

        pots_ntnn = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_ntnn_f = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn,pots_max_ntnn_f)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)
        
        el_ntnn_1 = np.inner(self.edge_index_ntnn.T, y[::2])
        el_ntnn_2 = np.inner(self.edge_index_ntnn.T, y[1::2])


        J_ntnn[::2] += self.ntnn_weight * ( np.inner(self.edge_index_ntnn, (1-phi_ntnn).T * pots_der_ntnn.T *  el_ntnn_1.T*1/3  )
                       +np.inner(self.edge_index_ntnn, phi_ntnn.T * self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1*1/3, 0).T) ) 
        J_ntnn[1::2] += self.ntnn_weight * ( np.inner(self.edge_index_ntnn, (1-phi_ntnn).T * pots_der_ntnn.T *  el_ntnn_2.T*1/3 )
                     +np.inner(self.edge_index_ntnn, phi_ntnn * self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2*1/3, 0).T) )
        
        J_full = J + J_ntnn[self.free_nodes]
        return J_full

    
    def Jacobian_mat(self,y,phi,dist_hist,y_1, hist_edges, edge_index,y_full, y_1_alpha, phi_alpha, reshape = True):
        J_front = np.zeros((self.L*self.B*2,self.num_alpha))
        RJ_front = np.zeros((self.L*self.B*2,self.num_alpha))
        if reshape: 
            z = np.zeros((self.L*self.B*2,))
            z[self.dir_nodes] = y_full[self.dir_nodes]
            z[self.free_nodes] = y
            y = z.copy()
        
        n = self.L*self.B*2
        x = np.reshape(y,(self.L*self.B,2))
        x_1 = np.reshape(y_1,(self.L*self.B,2))
        
        x_1_alpha = np.reshape(y_1_alpha,(self.L*self.B*2,self.num_alpha))

        dist =  (np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2)
        maxdist = np.maximum(dist,dist_hist)

        pots = np.expand_dims(self.Lennard_jones(dist,self.pots), axis = 1)
        pots_1d = self.Lennard_jones(dist,self.pots)
        pots_max_f =self.Lennard_jones(dist_hist,self.pots)
        pots_max = self.softmax(pots,pots_max_f)

        pots_der = self.Lennard_jones_der(dist,self.pots)
        pots_der_max = self.Lennard_jones_der(dist_hist,self.pots)

        pots_1d_mat = self.Lennard_jones(dist,self.pots_og)
        pots_max_f_mat = self.Lennard_jones(dist_hist,self.pots_og)
        pots_der_mat = self.Lennard_jones_der(dist,self.pots_og)
        pots_alpha = np.zeros((self.num_edges,self.num_alpha))
        pots_der_alpha = np.zeros((self.num_edges,self.num_alpha))
        pots_max_f_alpha = np.zeros((self.num_edges,self.num_alpha))
        
        vecc = np.zeros((self.num_edges,self.num_alpha))
        onesss = np.zeros((self.num_edges,self.num_alpha))
        
        count = 0
        for entry in self.alpha_inds:
            onesss[entry,count] = 1
            vecc[entry,count] = self.alpha_vec[entry]
            
            pots_alpha[entry,count] = pots_1d_mat[entry]
            pots_der_alpha[entry,count] = pots_der_mat[entry]
            pots_max_f_alpha[entry,count] = pots_max_f_mat[entry]
            count +=1
            

        el_1 = np.expand_dims(np.inner(self.edge_index.T, y[::2]), axis = 1)
        el_2 = np.expand_dims(np.inner(self.edge_index.T, y[1::2]), axis = 1)
        
        if self.diss_type == 'KV':
            grad_app = []
            grads_list = []
            for kk in range(len(self.alpha_list)):
                grad_app.append(self.get_gradients_new(-y_1_alpha[:,kk]))
                grads_list.append( self.gradlist)


        if self.nu != 0:     
            if self.diss_type == 'L2':                
                
                RJ_front[::2,:] = self.nu/(self.T[-1]/self.n_time_steps_og) * (- y_1_alpha[::2,:])
                RJ_front[1::2,:] = self.nu/(self.T[-1]/self.n_time_steps_og) * (- y_1_alpha[1::2,:])
            elif self.diss_type == 'KV': ### still to do
                for kk in range(len(self.alpha_list)):
                    RJ_front[::2,kk] = self.nu/(self.T[-1]/self.n_time_steps_og) * np.inner(self.gradlist.T, grad_app[kk][:self.gradlist.shape[0]].T)
                    RJ_front[1::2,kk] = self.nu/(self.T[-1]/self.n_time_steps_og) * np.inner(self.gradlist.T, grad_app[kk][self.gradlist.shape[0]:].T)

            else:
                print('assign dissipation type')
        phi = np.expand_dims(phi,axis = 1)
        onesies = np.ones(phi.shape)

        
        pots_der = np.expand_dims(pots_der, axis = 1)
        pots_max_f = np.expand_dims(pots_max_f, axis = 1)

        
        J_front[::2,:] = ( 
                           np.inner( self.edge_index,
                                   (1-phi).T * ( pots_der_alpha.T * el_1.T )
                                  )

                           
                           + np.inner(self.edge_index,
                                  phi.T* self.softmax_hess(pots.T, pots_max_f.T, pots_der.T* el_1.T, 0, pots_alpha.T, pots_max_f_alpha.T, pots_der_alpha.T * el_1.T, 0 )
                                  )

                         ) 
        J_front[1::2,:] = ( 
                              np.inner( self.edge_index,
                                    (1-phi).T * ( pots_der_alpha.T * el_2.T)
                                    )


                            + np.inner(self.edge_index,
                                  phi.T * self.softmax_hess(pots.T,pots_max_f.T, pots_der.T* el_2.T, 0, pots_alpha.T, pots_max_f_alpha.T, pots_der_alpha.T * el_2.T, 0 )
                                  )

                          )
        J_front[::2,:] += ( np.inner(self.edge_index,(1-phi_alpha).T * pots_der.T * el_1.T  )
                            +np.inner(self.edge_index, phi_alpha.T * self.softmax_der(pots, pots_max_f, pots_der * el_1, 0).T) )
        J_front[1::2,:] += ( np.inner(self.edge_index,(1-phi_alpha).T * pots_der.T * el_2.T  )
                            +np.inner(self.edge_index, phi_alpha.T * self.softmax_der(pots, pots_max_f, pots_der * el_2, 0).T) )


        J_front += RJ_front

        return J_front[self.free_nodes,:]#


    def Jacobian_ntnn_mat(self,y,phi,dist_hist,y_1, hist_edges, edge_index,phi_ntnn,dist_hist_ntnn, hist_edges_ntnn, edge_index_ntnn, y_full,y_1_alpha, phi_alpha, phi_ntnn_alpha):
        J_ntnn_front = np.zeros((self.L*self.B*2,self.num_alpha))

        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()
        
        J_front = self.Jacobian_mat(y, phi, dist_hist, y_1, hist_edges, edge_index, y_full, y_1_alpha, phi_alpha, reshape = False)
        
        x = np.reshape(y,(self.L*self.B,2))
        x_1_alpha = np.reshape(y_1_alpha,(self.L*self.B*2,self.num_alpha))
        
        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)
        maxdist = np.maximum(dist_ntnn,dist_hist_ntnn)

        pots_ntnn_1d = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_ntnn = np.expand_dims(pots_ntnn_1d, axis = 1)
        pots_max_ntnn_f = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn,pots_max_ntnn_f)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)
        pots_der_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))

        pots_der_ntnn_mat = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn_og)
        pots_ntnn_1d_mat = self.Lennard_jones(dist_ntnn,self.pots_ntnn_og)
        pots_max_ntnn_f_mat = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn_og)
        
        pots_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))
        pots_max_ntnn_f_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))
        count = 0

        vecc = np.zeros((self.num_edges_ntnn,self.num_alpha))
        onesss = np.zeros((self.num_edges_ntnn,self.num_alpha))
        for entry in self.alpha_inds_ntnn:
            onesss[entry,count] = 1
            vecc[entry,count] = self.alpha_vec_ntnn[entry]


            pots_der_ntnn_alpha[entry,count] = pots_der_ntnn_mat[entry]
            pots_ntnn_alpha[entry,count] = pots_ntnn_1d_mat[entry]
            pots_max_ntnn_f_alpha[entry,count] = pots_max_ntnn_f_mat[entry]
            count +=1

        pots_hess_ntnn = self.Lennard_jones_hess(dist_hist_ntnn,self.pots_ntnn)
        
        el_ntnn_1 = np.expand_dims(np.inner(self.edge_index_ntnn.T, y[::2]), axis = 1)
        el_ntnn_2 = np.expand_dims(np.inner(self.edge_index_ntnn.T, y[1::2]), axis = 1)

        phi_ntnn = np.expand_dims(phi_ntnn,axis = 1)

        pots_der_ntnn = np.expand_dims(pots_der_ntnn, axis = 1)
        pots_max_ntnn_f = np.expand_dims(pots_max_ntnn_f, axis = 1)



        onesies = np.ones(phi_ntnn.shape)
        J_ntnn_front[::2,:] += self.ntnn_weight *(
                                np.inner( self.edge_index_ntnn,
                                   (1-phi_ntnn).T * ( pots_der_ntnn_alpha.T * el_ntnn_1.T* 1/3 )
                                  )
                                + np.inner(self.edge_index_ntnn,
                                  phi_ntnn.T * self.softmax_hess(pots_ntnn.T, pots_max_ntnn_f.T, pots_der_ntnn.T * el_ntnn_1.T *1/3, 0, pots_ntnn_alpha.T, pots_max_ntnn_f_alpha.T, pots_der_ntnn_alpha.T * el_ntnn_1.T*1/3, 0 )
                                  )

                         )


        J_ntnn_front[1::2,:] += self.ntnn_weight *(
                                np.inner( self.edge_index_ntnn,
                                    (1-phi_ntnn).T * ( pots_der_ntnn_alpha.T * el_ntnn_2.T * 1/3)
                                    )
                                + np.inner(self.edge_index_ntnn,
                                  phi_ntnn.T * self.softmax_hess(pots_ntnn.T, pots_max_ntnn_f.T, pots_der_ntnn.T * el_ntnn_2.T*1/3 , 0, pots_ntnn_alpha.T, pots_max_ntnn_f_alpha.T, pots_der_ntnn_alpha.T * el_ntnn_2.T *1/3, 0 )
                                  )
                          )
        J_ntnn_front[::2,:] += self.ntnn_weight * ( np.inner(self.edge_index_ntnn, (1-phi_ntnn_alpha).T * pots_der_ntnn.T *  el_ntnn_1.T*1/3  )
                                 +np.inner(self.edge_index_ntnn, phi_ntnn_alpha.T * self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1*1/3, 0).T) ) 
        J_ntnn_front[1::2,:] += self.ntnn_weight * ( np.inner(self.edge_index_ntnn, (1-phi_ntnn_alpha).T * pots_der_ntnn.T *  el_ntnn_2.T*1/3 )
                                 +np.inner(self.edge_index_ntnn, phi_ntnn_alpha.T * self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2*1/3, 0).T) )
        J_front_full = J_front + J_ntnn_front[self.free_nodes,:]
        return J_front_full
    
    def hessian(self,y,phi,dist_hist,y_1, hist_edges,edge_index,y_full, reshape = True):
        H11 = spp.csr_array(np.zeros((self.L * self.B*2, self.L * self.B * 2)))
        R =  spp.csr_array(np.zeros((self.L * self.B*2, self.L * self.B * 2)))
        
        n = self.L*self.B*2
        if reshape: 
            z = np.zeros((self.L*self.B*2,))
            z[self.dir_nodes] = y_full[self.dir_nodes]
            z[self.free_nodes] = y
            y = z.copy()
        
        x = np.reshape(y,(self.L*self.B,2))

        dist =  (np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2)
        maxdist = np.maximum(dist,dist_hist)

        pots = self.Lennard_jones(dist,self.pots)
        pots_max_f = self.Lennard_jones(dist_hist,self.pots)
        pots_max = self.softmax(pots,pots_max_f)
        pots_der = self.Lennard_jones_der(dist,self.pots)
        pots_hess = self.Lennard_jones_hess(dist,self.pots)

        el_1 = np.inner(self.edge_index.T, y[::2])
        el_2 = np.inner(self.edge_index.T, y[1::2])

        soft_hess_1 = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_1**2 + 2 * pots_der, 0)
        soft_hess_2 = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_2**2 + 2 * pots_der, 0)
        soft_hess_12 = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_2*el_1, 0)

        soft_hess_1 = self.softmax_hess(pots, pots_max_f, pots_der * el_1, 0, pots_der * el_1, 0, pots_hess * 4 * el_1**2 + 2 * pots_der, 0)
        soft_hess_2 = self.softmax_hess(pots, pots_max_f, pots_der * el_2, 0, pots_der * el_2, 0, pots_hess * 4 * el_2**2 + 2 * pots_der, 0)
        soft_hess_12 = self.softmax_hess(pots, pots_max_f, pots_der * el_1, 0, pots_der * el_2, 0, pots_hess * 4 * el_2*el_1, 0)
        soft_hess_21 = self.softmax_hess(pots, pots_max_f, pots_der * el_2, 0, pots_der * el_1, 0,pots_hess * 4 * el_2*el_1, 0)

        if self.diss_type == 'KV':
            grad_app = self.get_gradients_new(y-y_1)
            grads_list = self.gradlist

        diag1 = range(0,self.L * self.B*2,2)
        diag2 = range(1,self.L * self.B*2,2)
        diag = range(self.L * self.B*2)

        abs_index = abs(self.edge_index)

        if self.diss_type == 'L2' and self.nu != 0: 
            R[diag,diag] = self.nu/(self.T[-1]/self.n_time_steps_og)
            
        H11[diag1,diag1] += 1/2*( np.inner(abs_index, (1-phi).T * (pots_hess.T * 4 * el_1.T**2 + 2 * pots_der.T))
                             + np.inner(abs_index, phi.T * soft_hess_1.T ) )
        H11[diag2,diag2] += 1/2*( np.inner(abs_index, (1-phi).T * (pots_hess.T * 4 * el_2.T**2 + 2 * pots_der.T))
                                 + np.inner(abs_index, phi.T * soft_hess_2.T) )
        H11[diag1,diag2] += 1/2*( np.inner( abs_index, (1-phi).T * pots_hess.T * 4 * el_2.T * el_1.T)
                               + np.inner( abs_index, phi.T * soft_hess_12.T) )
        H11[diag2,diag1] += 1/2*( np.inner( abs_index, (1-phi).T * pots_hess.T * 4 * el_2.T* el_1.T)
                               + np.inner( abs_index, phi.T * soft_hess_21.T) )
                
        if self.diss_type == 'KV' and self.nu != 0:
            for edge in range(grads_list.shape[0]):
                index = np.where(abs(grads_list[edge,:]) > 0)[0]
                R[np.ix_(2*index,2*index)] += self.nu/(self.T[-1]/self.n_time_steps_og) * np.expand_dims(grads_list[edge,index],axis =1).T* np.expand_dims(grads_list[edge,index],axis =1)
                R[np.ix_(2*index+1,2*index +1)] += self.nu/(self.T[-1]/self.n_time_steps_og) * np.expand_dims(grads_list[edge,index],axis =1).T* np.expand_dims(grads_list[edge,index],axis =1)


        ### assemble H11 asymmetric terms
        soft_hess_1_ass = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_1**2 - 2 * pots_der, 0)
        soft_hess_2_ass = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_2**2 - 2 * pots_der, 0)
        soft_hess_12_ass = self.softmax_der(pots, pots_max_f, pots_hess * 4 * el_2*el_1, 0)

        soft_hess_1_ass = self.softmax_hess(pots, pots_max_f, pots_der * el_1, 0, pots_der * el_1, 0,pots_hess * 4 * el_1**2 - 2 * pots_der, 0)
        soft_hess_2_ass = self.softmax_hess(pots, pots_max_f, pots_der * el_2, 0, pots_der * el_2, 0,pots_hess * 4 * el_2**2 - 2 * pots_der, 0)
        soft_hess_12_ass = self.softmax_hess(pots, pots_max_f, pots_der * el_1, 0,pots_der * el_2, 0,pots_hess * 4 * el_2*el_1, 0)
        soft_hess_21_ass = self.softmax_hess(pots, pots_max_f, pots_der * el_2, 0,pots_der * el_1, 0,pots_hess * 4 * el_2*el_1, 0)

        k,kj = np.where(abs(self.edge_index).T >0)
        k1 = 2*kj[::2]
        k2 = 2*kj[1::2]
        k = k[::2]

        H11[k1,k2] += 1/2*( (1-phi[k])* (pots_hess[k] * (-4)*el_1[k]**2 - 2*pots_der[k])
                           + (phi[k] * soft_hess_1_ass[k]) )
        H11[k1+1,k2+1] += 1/2*(  (1-phi[k])* (pots_hess[k] * (-4)*el_2[k]**2 - 2*pots_der[k])
                           + (phi[k]* soft_hess_2_ass[k]) )
        H11[k1+1,k2] += 1/2*( ( (1-phi[k]) * pots_hess[k] * (-4)*el_2[k]*el_1[k])
                       + (phi[k] * soft_hess_21_ass[k]) )
        H11[k1,k2+1] += 1/2*( ( (1-phi[k]) * pots_hess[k] * (-4)*el_2[k]*el_1[k])
                       + (phi[k] * soft_hess_12_ass[k]) )
        H11[k2,k1]   += H11[k1,k2]
        H11[k2+1,k1+1] += H11[k1+1,k2+1]
        H11[k2+1,k1] += H11[k1,k2+1]
        H11[k2,k1+1] += H11[k1+1,k2]

        ### extract relevant entries
        R = R[np.ix_(self.free_nodes,self.free_nodes)]
        H = H11[np.ix_(self.free_nodes,self.free_nodes)] + R

                
        return H

    def hessian_ntnn(self,y,phi,dist_hist,y_1, hist_edges,edge_index,phi_ntnn,dist_hist_ntnn, hist_edges_ntnn, edge_index_ntnn, y_full):
        H_ntnn = spp.csr_array(np.zeros((self.L * self.B*2, self.L * self.B * 2)))
        
        n = self.L*self.B*2
        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()

        H = self.hessian(y,phi,dist_hist,y_1, hist_edges,edge_index,y_full, reshape = False)

        x = np.reshape(y,(self.L*self.B,2))
        
        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)
        maxdist = np.maximum(dist_ntnn,dist_hist_ntnn)
        maxdist_ntnn = np.maximum(dist_ntnn,dist_hist_ntnn)

        pots_ntnn = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_ntnn_f = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn,pots_max_ntnn_f)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)

        pots_hess_ntnn = self.Lennard_jones_hess(dist_ntnn,self.pots_ntnn)

        el_ntnn_1 = np.inner(self.edge_index_ntnn.T, y[::2])
        el_ntnn_2 = np.inner(self.edge_index_ntnn.T, y[1::2])

        soft_hess_1 = self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_hess_ntnn * 4 * el_ntnn_1**2 *1/9 + 2 * pots_der_ntnn/3, 0)
        soft_hess_2 = self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_hess_ntnn * 4 * el_ntnn_2**2 *1/9 + 2 * pots_der_ntnn/3, 0)
        soft_hess_12 = self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_hess_ntnn * 4 * el_ntnn_2*el_ntnn_1/9, 0)

        soft_hess_1 = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1/3, 0, pots_der_ntnn * el_ntnn_1/3, 0, pots_hess_ntnn * 4 * el_ntnn_1**2/9 + 2 * pots_der_ntnn/3, 0)
        soft_hess_2 = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2/3, 0, pots_der_ntnn * el_ntnn_2/3, 0, pots_hess_ntnn * 4 * el_ntnn_2**2/9 + 2 * pots_der_ntnn/3, 0)
        soft_hess_12 = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1/3, 0, pots_der_ntnn * el_ntnn_2/3, 0, pots_hess_ntnn * 4 * el_ntnn_2*el_ntnn_1/9, 0)
        soft_hess_21 = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2/3, 0, pots_der_ntnn * el_ntnn_1/3, 0,pots_hess_ntnn * 4 * el_ntnn_2*el_ntnn_1/9, 0)

        diag1 = range(0,self.L * self.B*2,2)
        diag2 = range(1,self.L * self.B*2,2)
        diag = range(self.L * self.B*2)

        abs_index_ntnn = abs(self.edge_index_ntnn)
        
        H_ntnn[diag1,diag1] += self.ntnn_weight * 1/2*( np.inner(abs_index_ntnn, (1-phi_ntnn).T * (pots_hess_ntnn * 4 * el_ntnn_1**2 *1/9 + 2 * pots_der_ntnn/3).T )
                                                        + np.inner(abs_index_ntnn, phi_ntnn.T * soft_hess_1.T ) )
        
        H_ntnn[diag2,diag2] += self.ntnn_weight * 1/2*( np.inner(abs_index_ntnn, (1-phi_ntnn).T * (pots_hess_ntnn * 4 * el_ntnn_2**2 * 1/9 + 2 * pots_der_ntnn/3).T )
                                 + np.inner(abs_index_ntnn, phi_ntnn.T * soft_hess_2.T) )
        H_ntnn[diag1,diag2] += self.ntnn_weight * 1/2*( np.inner( abs_index_ntnn, (1-phi_ntnn).T * pots_hess_ntnn.T * 4 * el_ntnn_2.T * el_ntnn_1.T * 1/9)
                               + np.inner(abs_index_ntnn, phi_ntnn.T * soft_hess_12.T) )
        H_ntnn[diag2,diag1] += self.ntnn_weight * 1/2*( np.inner( abs_index_ntnn, (1-phi_ntnn).T * pots_hess_ntnn.T * 4 * el_ntnn_2.T * el_ntnn_1.T *1/9)
                               + np.inner(abs_index_ntnn, phi_ntnn.T * soft_hess_21.T) )
        
        ### assemble H11 asymmetric terms

        soft_hess_1_ass = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1/3, 0, pots_der_ntnn * el_ntnn_1/3, 0,pots_hess_ntnn * 4 * el_ntnn_1**2/9 - 2 * pots_der_ntnn/3, 0)
        soft_hess_2_ass = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2/3, 0, pots_der_ntnn * el_ntnn_2/3, 0,pots_hess_ntnn * 4 * el_ntnn_2**2/9 - 2 * pots_der_ntnn/3, 0)
        soft_hess_12_ass = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1/3, 0,pots_der_ntnn * el_ntnn_2/3, 0,pots_hess_ntnn * 4 * el_ntnn_2*el_ntnn_1/9, 0)
        soft_hess_21_ass = self.softmax_hess(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2/3, 0,pots_der_ntnn * el_ntnn_1/3, 0,pots_hess_ntnn * 4 * el_ntnn_2*el_ntnn_1/9, 0)

        k,kj = np.where(abs(self.edge_index_ntnn).T > 0)
        k1 = 2*kj[::2]
        k2 = 2*kj[1::2]
        k = k[::2]

        H_ntnn[k1,k2] += self.ntnn_weight * 1/2*( (1-phi_ntnn[k]) * (pots_hess_ntnn[k] * (-4)*el_ntnn_1[k]**2/9 - 2*pots_der_ntnn[k])
                       + (phi_ntnn[k] * soft_hess_1_ass[k]) )
        H_ntnn[k1+1,k2+1] += self.ntnn_weight * 1/2*( ((1-phi_ntnn[k]) *  (pots_hess_ntnn[k] * (-4)*el_ntnn_2[k]**2/9 - 2*pots_der_ntnn[k]))
                       + (phi_ntnn[k] *  soft_hess_2_ass[k]) )
        H_ntnn[k1+1,k2] += self.ntnn_weight * 1/2*( ( 1-phi_ntnn[k]) *  (pots_hess_ntnn[k] * (-4)*el_ntnn_2[k]*el_ntnn_1[k]/9)
                       + (phi_ntnn[k] * soft_hess_21_ass[k]) )
        H_ntnn[k1,k2+1] += self.ntnn_weight * 1/2*( ( 1-phi_ntnn[k]) *  (pots_hess_ntnn[k] * (-4)*el_ntnn_2[k]*el_ntnn_1[k]/9)
                       + (phi_ntnn[k] * soft_hess_12_ass[k]) )
        H_ntnn[k2,k1] += H_ntnn[k1,k2]
        H_ntnn[k2+1,k1+1] += H_ntnn[k1+1,k2+1]
        H_ntnn[k2+1,k1] += H_ntnn[k1,k2+1]
        H_ntnn[k2,k1+1] += H_ntnn[k1+1,k2]
        
        H_full = H + H_ntnn[np.ix_(self.free_nodes,self.free_nodes)]
        return H_full

    def full_hessian(self,y,phi,dist_hist,y_1, hist_edges,edge_index,y_full, reshape = True, return_parts = False):

        if reshape: 
            z = np.zeros((self.L*self.B*2,))
            z[self.dir_nodes] = y_full[self.dir_nodes]
            z[self.free_nodes] = y
            y = z.copy()
        
        ass_hessian = spp.csr_array(np.zeros((self.L * self.B*2 + self.num_edges, self.L * self.B * 2 + self.num_edges)))
        hessian_yy = self.hessian(y,phi,dist_hist,y_1, hist_edges,edge_index,y_full, reshape = False)
        hessian_yphi = spp.csr_array(np.zeros((self.num_edges, self.L * self.B * 2)))
        
        n = self.L*self.B*2
        x = np.reshape(y,(self.L*self.B,2))
        x_1 = np.reshape(y_1,(self.L*self.B,2))

        dist =  (np.inner(self.edge_index.T, y[::2])**2 + np.inner(self.edge_index.T, y[1::2])**2)
        maxdist = np.maximum(dist,dist_hist)

        pots = self.Lennard_jones(dist,self.pots)
        pots_max_f = self.Lennard_jones(dist_hist,self.pots)
        pots_max = self.softmax(pots,pots_max_f)

        pots_der = self.Lennard_jones_der(dist,self.pots)

        el_1 = np.inner(self.edge_index.T, y[::2])
        el_2 = np.inner(self.edge_index.T, y[1::2])
                
        hessian_yphi[np.ix_(range(self.num_edges), range(0,self.B*self.L*2,2))] = ( - np.inner(self.edge_index, pots_der.T * el_1.T  )
                                                                           + np.inner(self.edge_index, self.softmax_der(pots, pots_max_f, pots_der * el_1, 0).T) ) 
        hessian_yphi[np.ix_(range(self.num_edges), range(1,self.B*self.L*2,2))] = ( - np.inner( self.edge_index, pots_der.T *  el_2.T)
                                                                           + np.inner(self.edge_index , self.softmax_der(pots, pots_max_f, pots_der * el_2, 0).T) )
        if return_parts:
            return hessian_yy, hessian_yphi
        else:

            ass_hessian[np.ix_(self.free_nodes,self.free_nodes)] = hessian_yy
            ass_hessian[self.free_nodes, self.L * self.B*2:] = hessian_yphi[:,self.free_nodes].T
            ass_hessian[self.L * self.B*2:,self.free_nodes] = hessian_yphi[:,self.free_nodes]
            
            return ass_hessian

    def full_hessian_ntnn(self,y,phi,dist_hist,y_1, hist_edges,edge_index,phi_ntnn,dist_hist_ntnn, hist_edges_ntnn, edge_index_ntnn, y_full):

        z = np.zeros((self.L*self.B*2,))
        z[self.dir_nodes] = y_full[self.dir_nodes]
        z[self.free_nodes] = y
        y = z.copy()

        H_front_yy, H_front_yphi = self.full_hessian(y,phi,dist_hist,y_1, hist_edges,edge_index,y_full, reshape = False, return_parts = True)

        ass_hessian_ntnn = spp.csr_array(np.zeros((self.L * self.B*2 + self.num_edges + self.num_edges_ntnn, self.L * self.B * 2 + self.num_edges + self.num_edges_ntnn )))

        hessian_ntnn_yy = self.hessian_ntnn(y[self.free_nodes],phi,dist_hist,y_1, hist_edges,edge_index,phi_ntnn,dist_hist_ntnn, hist_edges_ntnn, edge_index_ntnn, y_full)
        hessian_ntnn_yphi = spp.csr_array(np.zeros((self.num_edges_ntnn, self.L * self.B * 2)))
        
        n = self.L*self.B*2
        x = np.reshape(y,(self.L*self.B,2))
        
        dist_ntnn = 1/3 * (np.inner(self.edge_index_ntnn.T, y[::2])**2 + np.inner(self.edge_index_ntnn.T, y[1::2])**2)
        maxdist = np.maximum(dist_ntnn,dist_hist_ntnn)

        pots_ntnn = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_ntnn_f = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn,pots_max_ntnn_f)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)
        
        el_ntnn_1 = np.inner(self.edge_index_ntnn.T, y[::2])
        el_ntnn_2 = np.inner(self.edge_index_ntnn.T, y[1::2])

        hessian_ntnn_yphi[np.ix_(range(self.num_edges_ntnn), range(0,self.B*self.L*2,2))] += self.ntnn_weight * ( - np.inner(self.edge_index_ntnn,  pots_der_ntnn.T *  el_ntnn_1.T*1/3  )
                                                                                                                    +np.inner(self.edge_index_ntnn, self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_1*1/3, 0).T) ) 
        hessian_ntnn_yphi[np.ix_(range(self.num_edges_ntnn), range(1,self.B*self.L*2,2))] += self.ntnn_weight * ( - np.inner(self.edge_index_ntnn, pots_der_ntnn.T *  el_ntnn_2.T*1/3 )
                                                                                                                    +np.inner(self.edge_index_ntnn, self.softmax_der(pots_ntnn, pots_max_ntnn_f, pots_der_ntnn * el_ntnn_2*1/3, 0).T) )
        

        ass_hessian_ntnn[np.ix_(self.free_nodes,self.free_nodes)] = hessian_ntnn_yy + H_front_yy
        ass_hessian_ntnn[self.free_nodes,self.L * self.B*2:self.L * self.B*2 + self.num_edges] = H_front_yphi[:,self.free_nodes].T
        ass_hessian_ntnn[self.L * self.B*2:self.L * self.B*2 + self.num_edges,self.free_nodes] = H_front_yphi[:,self.free_nodes]
        ass_hessian_ntnn[self.free_nodes,self.L * self.B*2 + self.num_edges:self.L * self.B*2 + self.num_edges + self.num_edges_ntnn] = hessian_ntnn_yphi[:,self.free_nodes].T
        ass_hessian_ntnn[self.L * self.B*2 + self.num_edges: self.L * self.B*2 + self.num_edges + self.num_edges_ntnn, self.free_nodes] = hessian_ntnn_yphi[:,self.free_nodes]
        
        return ass_hessian_ntnn
    
        
    def solve(self, method):
        print('################## starting solving ####################')
        zeitanfang = time.time()

        current_time = []
        current_time.append(self.basetime[0])
        current_time.append(self.basetime[1])
        j_end = 0

        ### setting up the folder to store in ( this lets you assign a personal id)
        if self.save_output:
            self.folder = './2dsprings/' + str(self.id)
            print(self.folder)
            if not os.path.exists(self.folder): 
                os.makedirs(self.folder)

            filename = self.folder + '/meta.npy'
            with open(filename, 'wb') as f:
                np.savez(f, ntnn = self.ntnn, L = self.L, B = self.B, max_steps = self.n_time_steps)

        ### base configuration
        X = self.get_vertices()
        self.X = X.reshape(self.L*self.B,2).copy()
        self.Y[:,0] = np.reshape(self.X.copy(),(self.L*self.B*2)).copy()
        self.Z = self.Y[:,0].copy()


        ### generate dirichlet nodes
        nodes = self.get_dirichlet_nodes()
        self.dir_nodes = np.concatenate(np.asarray(nodes))
        full_set = range(self.L * self.B*2)
        self.free_nodes = np.setdiff1d(full_set,self.dir_nodes)

        ### generate edges 
        edge_hist, edge_index= self.get_edges(self.X,ret_index = True)        
        self.edge_index = np.where(edge_index <2, edge_index, -1).copy() ### prime the edge lists
        self.edge0_sort = np.where(self.edge_index.T == 1)
        self.edge1_sort = np.where(self.edge_index.T == -1)
        if self.ntnn:
            edge_hist_ntnn, edge_index_ntnn = self.get_edges_ntnn(self.X,ret_index = True)
            self.edge_index_ntnn = np.where(edge_index_ntnn <2, edge_index_ntnn, -1).copy()
            self.edge0_sort_ntnn = np.where(self.edge_index_ntnn.T == 1)
            self.edge1_sort_ntnn = np.where(self.edge_index_ntnn.T == -1)

        ### apply material parameters
        if self.inhom_material:
            self.edge_hist_og = edge_hist.copy()
            self.apply_material_parameters(self.edge_hist_og)
            if self.ntnn:
                self.edge_hist_ntnn_og = edge_hist_ntnn.copy()
                self.apply_material_parameters_ntnn(self.edge_hist_ntnn_og)

        ### damage and history variables
        r_hist = self.get_distance(edge_hist)
        phi_hist = self.damage_var(r_hist) 
        dist_1 = r_hist.copy()
        if self.ntnn:
            r_hist_ntnn = self.get_distance(edge_hist_ntnn) * 1/3
            phi_hist_ntnn = self.damage_var_ntnn(r_hist_ntnn) 
            dist_1_ntnn = r_hist_ntnn.copy()
            self.hist_ntnn_list = []
            self.hist_ntnn_list.append(dist_1_ntnn)
        

        if self.inhom_material and self.compute_mat_der:
            x_1_alpha = self.Y_alpha[:,:,0].copy()
            r_hist_alpha = np.zeros((self.num_edges, self.num_alpha))
            self.phi_hist_alpha = self.damage_var_mat(r_hist, r_hist_alpha)
            dist_1_alpha = r_hist_alpha.copy()
            self.hist_list_alpha = []
            self.hist_list_alpha.append(dist_1_alpha)

            if self.ntnn:
                r_hist_ntnn_alpha = np.zeros((self.num_edges_ntnn, self.num_alpha))
                self.phi_hist_ntnn_alpha = self.damage_var_ntnn_mat(r_hist_ntnn,r_hist_ntnn_alpha) 
                dist_1_ntnn_alpha = r_hist_ntnn_alpha.copy()
                self.hist_ntnn_list_alpha = []
                self.hist_ntnn_list_alpha.append(dist_1_ntnn_alpha)
        
        ### generate the triangle and gradient lists once
        [grad_app, grad_list] = self.get_gradients_new(self.X, Return_Flag = True)
        
        ### tracking eigenvalue
        self.eigs = np.zeros((0,))
        self.eigsn = np.zeros((self.num_eigs,self.Y.shape[1]))
        if self.compute_mat_der or self.track_eigs:
            if self.ntnn:
                H = self.hessian_ntnn(self.Y[self.free_nodes,0],phi_hist,r_hist, self.Y[:,0].copy(), edge_hist, edge_index, phi_hist_ntnn,r_hist_ntnn, edge_hist_ntnn, edge_index_ntnn, self.Y[:,0])
            else: 
                H = self.hessian(self.Y[self.free_nodes,0],phi_hist,r_hist, self.Y[:,0].copy(), edge_hist, edge_index, self.Y[:,0])

        if self.track_eigs:
            (eigss,eigsvec)=eigs(H,k= self.num_eigs,which='SR')
            self.eigs = np.append(self.eigs,min(eigss))
            self.eigsn[:,0] = eigss
            
        self.hist_list = []
        self.hist_list.append(dist_1)
        x_1 = self.Y[:,0]

        ### list of jacobians
        self.J_list = []
        self.J_list_alpha = []
                       

        ### initialize timestep counter
        i = 1
        ii = 1 ### index of where i am in time, c
        min_count = 0


        ### type of energy functional ( nearest neighbour edges only, next to nearest neighbour edges added)
        if self.ntnn:
            objective_energy = lambda x: self.energy_ntnn(x,phi_hist,r_hist,phi_hist_ntnn, r_hist_ntnn, self.Z)
            J_f = lambda x: self.Jacobian_ntnn(x,phi_hist,r_hist,x_1, edge_hist, edge_index,phi_hist_ntnn,r_hist_ntnn, edge_hist_ntnn, edge_index_ntnn, self.Z)
            H_f = lambda x: self.hessian_ntnn(x,phi_hist,r_hist,x_1, edge_hist,edge_index,phi_hist_ntnn,r_hist_ntnn, edge_hist_ntnn, edge_index_ntnn, self.Z)
            if self.compute_mat_der:
                objective_energy_mat = lambda x,x_mat: self.energy_ntnn_mat(x, phi_hist, r_hist, phi_hist_ntnn, r_hist_ntnn, self.Z,
                                                                            x_mat, self.phi_hist_alpha, r_hist_alpha, self.phi_hist_ntnn_alpha, r_hist_ntnn_alpha)
        else: 
            objective_energy = lambda x: self.energy(x,phi_hist,r_hist,self.Z)
            J_f = lambda x: self.Jacobian(x,phi_hist,r_hist,x_1, edge_hist, edge_index,self.Z)
            H_f = lambda x: self.hessian(x,phi_hist,r_hist,x_1, edge_hist,edge_index,self.Z)
            if self.compute_mat_der:
                objective_energy_mat = lambda x,x_mat: self.energy_mat(x, phi_hist, r_hist, self.Z, x_mat, self.phi_hist_alpha, r_hist_alpha)

        ### type of dissipation (L2 norm or Kelvin Voigt)
        if self.diss_type == 'L2': 
            objective_function = lambda x: objective_energy(x) + self.L2_dissipation(x,x_1,self.Z)
            objective_diss = lambda x: self.L2_dissipation(x,x_1,self.Z)
            if self.compute_mat_der:
                objective_function_mat = lambda x,x_mat: objective_energy_mat(x,x_mat) + self.L2_dissipation_mat(x,x_1,self.Z, x_mat, x_1_alpha)
            
        elif self.diss_type == 'KV':
            objective_function = lambda x: objective_energy(x) + self.KV_dissipation(x,x_1,self.Z)
            objective_diss = lambda x: self.KV_dissipation(x,x_1,self.Z)
            grad_app, self.grads_list = self.get_gradients_new(x_1, Return_Flag = True)



        #### tracking the energy
        self.energies = []
        self.objectives = []
        self.energies.append( objective_energy(self.Y[self.free_nodes,0]) )
        self.objectives.append( objective_function(self.Y[self.free_nodes,0]) )

        if self.compute_mat_der:
            self.energies_mat = []
            self.objectives_mat = []
            self.energies_mat.append( objective_energy_mat(self.Y[self.free_nodes,0],self.Y_alpha[self.free_nodes,:,0]) )
            self.objectives_mat.append( objective_function_mat(self.Y[self.free_nodes,0], self.Y_alpha[self.free_nodes,:,0]) )
        
        SMALL_STEPS = False
        KEEP_ITERATING = True
        while KEEP_ITERATING:
            PROCEED = True

            if self.loading_spatial_dep:
                for j,entry in enumerate(self.boundary):
                    for k,loadstep in enumerate(self.loading[j]):
                        self.Z[nodes[j][k::2]] = self.Y[nodes[j][k::2],0] + loadstep(current_time[-1], self.Y[nodes[j],0])
            else:
                for j,entry in enumerate(self.boundary):
                    for k,loadstep in enumerate(self.loading[j]):
                        self.Z[nodes[j][k::2]] = self.Y[nodes[j][k::2],0] + loadstep(current_time[-1])

            if self.print_progress:
                print('----------- Computing timestep %i -------------' %(i))

            ### boundary constraints for the minimization problem updated in timestep i

            if method == 'one step':

                ### solve
                zeitopt = time.time()
                res = spopt.minimize(
                    objective_function,
                    self.Y[self.free_nodes,i-1],
                    jac = J_f,
                    hess = H_f,
                    method = 'L-BFGS-B'
                    )
                endopt = time.time() - zeitopt
                
                self.J_list.append(J_f(res['x']))
                
                ### update tracking variables and solution
                self.Y[self.free_nodes,i] = res['x'].copy()
                self.Y[self.dir_nodes,i] = self.Z[self.dir_nodes]



                ### hessian
                if self.compute_mat_der or self.track_eigs:
                    if self.ntnn: 
                        H = self.hessian_ntnn(res['x'],phi_hist,r_hist, self.Y[:,i-1].copy(), edge_hist, edge_index, phi_hist_ntnn,r_hist_ntnn, edge_hist_ntnn, edge_index_ntnn, self.Z)
                    else:
                        H = self.hessian(res['x'],phi_hist,r_hist, self.Y[:,i-1].copy(), edge_hist, edge_index, self.Z)

                if self.track_eigs:
                    zeit1 = time.time()
                    (eigss,eigsvecs) = eigs(H, k = self.num_eigs, which='SR')
                    end1 = time.time() - zeit1
                if self.print_progress:
                    print('OPTIMIZATION TIME:',endopt)
                    print('EIGENVALUE EVALUATION TIME:',end1)
                    print('SIZE OF MINIMAL EIGENVALUE', min(eigss))
                    
                if current_time[-1] < self.T[-1]:    
                    if min(eigss) <= 0 and not SMALL_STEPS:
                        SMALL_STEPS = True
                        PROCEED = False
                        i = i-1
                        ii = ii-1
                        j_start = self.ref_fact*ii ## multiply only by 5-1, because i still at pre refinement value
                        min_count = 0
                        ii = j_start ### transform i forward to refinement
                        current_time = current_time[:-1]
                        current_time.append(self.finetime[ii+1])
                        if self.print_progress:
                            print('time:',self.finetime[ii] ,j_start,ii,i)#
                            print('next timestep:',self.finetime[ii+1],ii+1,i+1)
                            print('##--##--##--##--##--##--##--SMALLER STEPS NOW--##--##--##--##--##--##--##--##--##--##--##')
                    elif SMALL_STEPS and min(eigss) >= 0 and (ii)%self.ref_fact==0 and min_count >= self.ref_fact*3:
                        SMALL_STEPS = False
                        j_end = int(ii/self.ref_fact)
                        ii = j_end
                        current_time.append(self.basetime[ii+1])
                        if self.print_progress:
                            print('time:',self.basetime[ii],j_end,ii,i )
                            print('next timestep:',self.basetime[ii+1],ii+1,i+1)
                            print('++--++--++--++--++--++--++--LARGER STEPS NOW--++--++--++--++--++--++--++--++--++--++')
                    elif SMALL_STEPS:
                        min_count +=1
                        current_time.append(self.finetime[ii+1])
                        if self.print_progress:
                            print('time:',self.finetime[ii] ,j_start,i)
                            print('next timestep:',self.finetime[ii+1],ii,ii+1,i,i+1)
                            print('##--##--##--##--##--##--##--SMALL STEPS--##--##--##--##--##--##--##--##--##--##--##')
                    else: 
                        current_time.append(self.basetime[ii+1])
                        if self.print_progress:
                            print('time:',self.basetime[ii],j_end ,i)
                            print('next timestep:',self.basetime[ii+1],ii,ii+1,i,i+1)
                            print('++--++--++--++--++--++--++--LARGE STEPS--++--++--++--++--++--++--++--++--++--++--++')
                else:
                    print('############################# LAST STEP ##########################')
                    KEEP_ITERATING = False

                if PROCEED:
                    if self.track_eigs:
                        self.eigs = np.append(self.eigs,min(eigss))
                        self.eigsn[:,i] = eigss

                    if self.inhom_material and self.compute_mat_der:

                        doublecheck = False
                        alph_stepsize = 10**-6

                        if self.ntnn:
                            J_front = self.Jacobian_ntnn_mat(res['x'].copy(),phi_hist,r_hist,x_1, edge_hist, edge_index,phi_hist_ntnn,r_hist_ntnn, edge_hist_ntnn, edge_index_ntnn, self.Z, x_1_alpha, self.phi_hist_alpha, self.phi_hist_ntnn_alpha)
                        else:
                            J_front = self.Jacobian_mat(res['x'].copy(),phi_hist,r_hist,x_1, edge_hist, edge_index,self.Z, x_1_alpha, self.phi_hist_alpha, reshape = True)
                                    
                        x_alpha = np.zeros((len(self.free_nodes), self.num_alpha))
                        tik_nu = 10**-3
                        Left_matrix = spp.csr_array(np.zeros((self.L * self.B*2, self.L * self.B * 2)))
                        Left_matrix = np.inner(H,H) + tik_nu * spp.eye(len(self.free_nodes))
                        Right_matrix = np.inner(H.todense(),J_front.T)  + tik_nu * x_1_alpha[self.free_nodes,:]
                        x_alpha = spsolve(Left_matrix, Right_matrix)
                        #############################################################

                        ### update tracking variables and solution
                        self.Y_alpha[self.free_nodes,:,i] = x_alpha.copy()
                    
                    edge_1 = np.zeros((self.num_edges,2,2))
                    edge_1[self.edge0_sort[0],0,0] = self.Y[::2,i][self.edge0_sort[1]]
                    edge_1[self.edge0_sort[0],0,1] = self.Y[1::2,i][self.edge0_sort[1]]
                    edge_1[self.edge1_sort[0],1,0] = self.Y[::2,i][self.edge1_sort[1]]
                    edge_1[self.edge1_sort[0],1,1] = self.Y[1::2,i][self.edge1_sort[1]]
                    dist_1 = np.inner(self.edge_index.T, self.Y[::2,i])**2 + np.inner(self.edge_index.T, self.Y[1::2,i])**2
                    

                    max_dist = np.maximum(dist_1,r_hist)
                    max_list = np.where(dist_1 == max_dist)

                    edge_hist[max_list] = edge_1[max_list]
                    r_hist[max_list] = dist_1[max_list]
                    phi_hist = self.damage_var(r_hist)
                    self.hist_list.append(r_hist.copy())

                    if self.ntnn:
                        dist_1_ntnn = 1/3*( np.inner(self.edge_index_ntnn.T, self.Y[::2,i])**2 + np.inner(self.edge_index_ntnn.T, self.Y[1::2,i])**2)
                        
                        r_hist_ntnn = np.maximum(dist_1_ntnn, r_hist_ntnn)
                        max_list_ntnn = np.where(dist_1_ntnn == r_hist_ntnn)
                        phi_hist_ntnn = self.damage_var_ntnn(r_hist_ntnn)
                        self.hist_ntnn_list.append(r_hist_ntnn)
                        

                        

                    if self.inhom_material and self.compute_mat_der:
                        dist_1_alpha = 2* ( np.inner(self.edge_index.T, np.expand_dims(self.Y[::2,i], axis = 1).T * self.Y_alpha[::2,:,i].T)
                                           + np.inner(self.edge_index.T, np.expand_dims(self.Y[1::2,i], axis = 1).T * self.Y_alpha[1::2,:,i].T ) )
                        r_hist_alpha[max_list,:] = dist_1_alpha[max_list,:]
                        self.phi_hist_alpha = self.damage_var_mat(r_hist, r_hist_alpha)
                        self.hist_list_alpha.append(r_hist_alpha)
                        if self.ntnn:
                            dist_1_ntnn_alpha = 2/3* ( np.inner(self.edge_index_ntnn.T, np.expand_dims(self.Y[::2,i], axis = 1).T * self.Y_alpha[::2,:,i].T)
                                           + np.inner(self.edge_index_ntnn.T, np.expand_dims(self.Y[1::2,i], axis = 1).T * self.Y_alpha[1::2,:,i].T ) )
                            r_hist_ntnn_alpha[max_list_ntnn,:] = dist_1_ntnn_alpha[max_list_ntnn,:]
                            self.phi_hist_ntnn_alpha = self.damage_var_ntnn_mat(r_hist_ntnn, r_hist_ntnn_alpha)
                            self.hist_ntnn_list_alpha.append(r_hist_ntnn_alpha)

                        #### update energies
                        self.energies_mat.append( objective_energy_mat(self.Y[self.free_nodes,i],self.Y_alpha[self.free_nodes,:,i]) )
                        self.objectives_mat.append( objective_function_mat(self.Y[self.free_nodes,i], self.Y_alpha[self.free_nodes,:,i]) )
                        x_1_alpha = self.Y_alpha[:,:,i].copy()


                    #### update energies
                    self.energies.append( objective_energy(self.Y[self.free_nodes,i]) )
                    self.objectives.append( objective_function(self.Y[self.free_nodes,i]) )
                    x_1 = self.Y[:,i].copy()

            i+=1
            ii+=1
        self.time_needed = time.time()- zeitanfang
        print('TIME NEEDED:', self.time_needed)
        if self.save_output: 
            filename = self.folder + '/data.npy'
            with open(filename, 'wb') as f:
                if self.ntnn: 
                    np.savez(f, x = self.Y[:,:len(current_time)], eigsn = self.eigsn[:,:len(current_time)], eigs = self.eigs, T = current_time, energy = self.energies, objective = self.objectives, r_hist = self.hist_list, R_1 = self.R_1, R_2 = self.R_2, r_hist_ntnn = self.hist_ntnn_list,  R_1_ntnn = self.R_1_ntnn, R_2_ntnn = self.R_2_ntnn)
                if not self.ntnn:
                    np.savez(f, x = self.Y[:,:len(current_time)], eigsn = self.eigsn[:,:len(current_time)], eigs = self.eigs, T = current_time, energy = self.energies, objective = self.objectives, r_hist = self.hist_list, R_1 = self.R_1, R_2 = self.R_2)
            if self.inhom_material and self.compute_mat_der:
                filename_der = self.folder + '/data_der.npy'
                if self.ntnn: 
                    with open(filename_der, 'wb') as f:
                        np.savez(f,x_alpha = self.Y_alpha, energy_mat = self.energies_mat, objective_mat = self.objectives_mat, r_hist_alpha = self.hist_list_alpha, r_hist_ntnn_alpha = self.hist_ntnn_list_alpha, J_list = self.J_list, J_list_alpha = self.J_list_alpha)
                if not self.ntnn:
                    with open(filename_der, 'wb') as f:
                        np.savez(f,x_alpha = self.Y_alpha, energy_mat = self.energies_mat, objective_mat = self.objectives_mat, r_hist_alpha = self.hist_list_alpha, J_list = self.J_list, J_list_alpha = self.J_list_alpha)
        print('################## finished solving ####################')
                


            
        
