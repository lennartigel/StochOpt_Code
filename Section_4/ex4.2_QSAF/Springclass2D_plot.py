import numpy as np
import os as os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mycolorpy import colorlist as mcp
import scipy.optimize as spopt
from autograd import grad, hessian
import datetime
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import scipy.sparse as spp
from numpy import exp
import time as time


class SpringFracture:
    def __init__(self, L, B, n_time_steps, ntnn = False):
        ### base parameters
        self.pref_dist = 1
        self.L = L
        self.B = B

        ### default value whether to print updates on progress
        self.print_progress = False
        self.save_output = True

        ### time points
        self.n_time_steps = n_time_steps
        basetime = np.linspace(0,1,n_time_steps)
        self.T = basetime.copy()
        self.T_bu = basetime.copy()

        ### solution list
        self.Y = np.zeros((self.L*self.B*2,n_time_steps))

        ## variables for edge computation
        self.num_edges = self.L*(self.B-1) + self.B*(self.L-1) + 1*(self.L-1)*(self.B-1)
        self.s_1 = self.L*(self.B-1)
        self.s_2 = self.s_1 + self.B*(self.L-1)
        
        ### cutoff parameters for damage variable
        self.R_1 = 1.2*self.pref_dist * np.ones((self.num_edges,))
        self.R_2 = self.R_1.copy() *2.5 -self.pref_dist* np.ones((self.num_edges,))

        
        self.pots = np.ones((self.num_edges,))*-1

        self.standard_str = 1.2

        self.num_eigs = 2

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
            s = [1.2,1.2]
            if self.Friedrich_setup or self.Friedrich_setup_2: 
                self.alpha_fix_list = [1.2,1.2,1.2,1.2]
            self.num_alpha = 2
            self.Y_alpha = np.zeros((self.L*self.B*2,self.num_alpha, n_time_steps))
            self.PLOT = False

        ### softmax parameters for either softmax functions
        self.softmaxalpha = 5
        self.steepness = 1/1000
        self.phi_max_ex = True
        self.smoothmaxalpha = 10**-12

        self.smax = lambda a,b: (a+b+np.sqrt((a-b)**2 + 10**-12))/2
        self.smax_der = lambda a,b,dera,derb: (dera+derb + (a-b)*(dera-derb)/np.sqrt((a-b)**2 + 10**-12))/2

        ### storage container
        self.id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')


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
            #print(edge_index)
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
##        if np.any(self.R_1- self.R_2 !=0): 
        np.seterr(divide='ignore')
        phi = np.zeros(r.shape)

        if self.phi_max_ex: 
            ### original
            ### where damage in transition from 0 to 1
            phi = np.where(r > self.R_1**2, (r-self.R_1**2)/(self.R_2**2 - self.R_1**2), phi)
            ### where no damage
            phi = np.where(r - self.R_1**2 < 0, 0, phi)
            ### where damage is 1
            phi = np.where(r > self.R_2**2, 1, phi)
        else: 
            phi = -self.smax( -self.smax(0,(r-self.R_1**2)/(self.R_2**2-self.R_1**2) ), -1)
        
        return phi

    def damage_var_mat(self,r,r_mat):
        np.seterr(divide='ignore',invalid='ignore')
        phi_alpha = np.zeros(r_mat.shape)
        expansion = np.ones((self.num_edges, self.num_alpha))
        rr = np.expand_dims(r,axis = 1) * expansion
        RR_1 = np.expand_dims(self.R_1,axis = 1) * expansion
        RR_2 = np.expand_dims(self.R_2,axis = 1) * expansion

        if self.phi_max_ex: 
            ### original
            ### where damage in transition from 0 to 1
            phi_alpha = np.where(rr >= RR_1**2, r_mat/(RR_2**2 - RR_1**2), phi_alpha)
            ### where no damage
            phi_alpha = np.where(rr - RR_1**2 < 0, 0, phi_alpha)
            ### where damage is 1
            phi_alpha = np.where(rr > RR_2**2, 0, phi_alpha)
        else: 
            phi_alpha = -self.smax_der( -self.smax(0,(rr-self.RR_1**2)/(self.RR_2**2-self.RR_1**2)), -1, -self.smax_der(0,(rr-self.RR_1**2)/(self.RR_2**2-self.RR_1**2), 0, r_mat/(self.RR_2**2-self.RR_1**2)), 0)
        return phi_alpha

    def damage_var_ntnn(self,r):
##        if np.any(self.R_1_ntnn - self.R_2_ntnn !=0): 
        np.seterr(divide='ignore')
        phi = np.zeros(r.shape)
        if self.phi_max_ex: 
            ### where damage in transition from 0 to 1
            phi = np.where(r >= self.R_1_ntnn**2, (r-self.R_1_ntnn**2)/(self.R_2_ntnn**2 - self.R_1_ntnn**2), phi)
            ### where damage is 1
            phi = np.where(r - self.R_2_ntnn**2 >=0, 1, phi)
            ### where no damage
            phi = np.where(r - self.R_1_ntnn**2 < 0, 0, phi)
        else:
            phi = -self.smax( -self.smax(0,(r-self.R_1_ntnn**2)/(self.R_2_ntnn**2-self.R_1_ntnn**2) ), -1)
        
            
        return phi

    def damage_var_ntnn_mat(self,r,r_mat):
        np.seterr(divide='ignore',invalid='ignore')
        phi_alpha = np.zeros(r_mat.shape)
        expansion = np.ones((self.num_edges_ntnn, self.num_alpha))
        rr = np.expand_dims(r,axis = 1) * expansion
        RR_1 = np.expand_dims(self.R_1_ntnn,axis = 1) * expansion
        RR_2 = np.expand_dims(self.R_2_ntnn,axis = 1) * expansion
        if self.phi_max_ex: 
            ### where damage in transition from 0 to 1
            phi_alpha = np.where(rr >= RR_1**2, r_mat/(RR_2**2 - RR_1**2), phi_alpha)
            ### where damage is 1
            phi_alpha = np.where(rr >= RR_2**2, 0, phi_alpha)
            ### where no damage
            phi_alpha = np.where(rr - RR_1**2 < 0, 0, phi_alpha)
        else:
            phi_alpha = -self.smax_der( -self.smax(0,(rr-self.RR_1**2)/(self.RR_2**2-self.RR_1**2)), -1, -self.smax_der(0,(rr-self.RR_1**2)/(self.RR_2**2-self.RR_1**2), 0, r_mat/(self.RR_2**2-self.RR_1**2)), 0)
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
##                cond_set_base = np.setdiff1d(cond_set_base, np.where(self.X[:,0] <= self.pref_dist/2)[0])
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
##            ind = np.intersect1d(ind_left, ind_band)

            
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

        self.alpha_vec = np.ones((self.num_edges,))
        if self.Friedrich_setup or self.Friedrich_setup_2: 
            for k,entry in enumerate(self.alpha_fix_inds):
                self.alpha_vec[entry] = self.alpha_fix_list[k]
        for k,entry in enumerate(self.alpha_inds):
            self.alpha_vec[entry] = self.alpha_list[k]
        self.pots = self.alpha_vec* self.pots
        
        if self.PLOT: 
            fig,ax = plt.subplots(nrows = 1)
            for entry in edge_hist:
                ax.plot(entry[:,0], entry[:,1], zorder = 0,c='b')
            for entry in edge_hist[ind,:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 1, c = 'red')
            for entry in edge_hist[ind_2,:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 2, c = 'green')
            for entry in edge_hist[ind_3,:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 3, c = 'yellow')
            if not self.ntnn: 
                plt.show()
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
        self.alpha_vec_ntnn = np.ones((self.num_edges_ntnn,))

        if self.Friedrich_setup or self.Friedrich_setup_2: 
            for k,entry in enumerate(self.alpha_fix_inds_ntnn):
                self.alpha_vec_ntnn[entry] = self.alpha_fix_list[k]

        for k,entry in enumerate(self.alpha_inds_ntnn):
            self.alpha_vec_ntnn[entry] = self.alpha_list[k]

        self.pots_ntnn =  self.alpha_vec_ntnn * self.pots_ntnn

        if self.PLOT: 
            fig_ntnn,ax_ntnn = plt.subplots(nrows = 1)
            for entry in edge_hist_ntnn:
                ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 0, c = 'b')
            for entry in edge_hist_ntnn[ind,:,:]:
                ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 2, c = 'red')
            for entry in edge_hist_ntnn[ind_2,:,:]:
                ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 1, c = 'green')
            for entry in edge_hist_ntnn[ind_3,:,:]:
                ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 4, c = 'yellow')
            plt.show()
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
        dist_alpha = 2* ( np.inner(self.edge_index.T, np.expand_dims(y[::2], axis = 1).T * y_alpha[::2,:].T)
                                       + np.inner(self.edge_index.T, np.expand_dims(y[1::2], axis = 1).T * y_alpha[1::2,:].T ) )

        pots = np.expand_dims(self.Lennard_jones(dist,self.pots), axis = 1)        
        pots_1d = self.Lennard_jones(dist,self.pots)
        pots_max_f = self.Lennard_jones(dist_hist,self.pots)
        maxdist = np.maximum(dist,dist_hist)
        pots_max = self.softmax(pots_1d,pots_max_f)

        pots_der = self.Lennard_jones_der(dist,self.pots)
        pots_der_max = self.Lennard_jones_der(dist_hist,self.pots)

        pots_alpha = np.zeros((self.num_edges,self.num_alpha))
        pots_max_f_alpha = np.zeros((self.num_edges,self.num_alpha))

        count = 0
        for entry in self.alpha_inds:
            

            pots_alpha[entry,count] = pots_1d[entry]
            pots_max_f_alpha[entry,count] = pots_max_f[entry]
            count +=1

        eny_alpha = np.zeros(self.num_alpha)
        for ii in range(self.num_alpha):
            eny_alpha[ii] = 1/2*(   np.inner((1-phi),pots_der*dist_alpha[:,ii]) + np.inner(phi, self.softmax_der(pots_1d, pots_max_f, pots_der*dist_alpha[:,ii], pots_der_max * dist_hist_alpha[:,ii]) )
                                    + np.inner((1-phi),pots_alpha[:,ii]) + np.inner(phi,self.softmax_der(pots_1d, pots_max_f, pots_alpha[:,ii], pots_max_f_alpha[:,ii]) )
                                    + np.inner((1-phi_alpha[:,ii]),pots_1d) + np.inner(phi_alpha[:,ii],pots_max)
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
        diss = 1/2 * self.nu/(self.T[-1]/self.n_time_steps) * np.inner(x-y_1,x-y_1)
        return diss

    def KV_dissipation(self,y,y_1, y_full):
        x = np.zeros((self.L*self.B*2,))
        x[self.dir_nodes] = y_full[self.dir_nodes]
        x[self.free_nodes] = y
        grad_app = self.get_gradients_new(x-y_1)
        diss = 1/2 * self.nu/(self.T[-1]/self.n_time_steps) * ( np.inner(grad_app , grad_app ))
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
        dist_ntnn_alpha = 2/3* ( np.inner(self.edge_index_ntnn.T, np.expand_dims(y[::2], axis = 1).T * y_alpha[::2,:].T)
                                       + np.inner(self.edge_index_ntnn.T, np.expand_dims(y[1::2], axis = 1).T * y_alpha[1::2,:].T ) )

        pots_ntnn = np.expand_dims(self.Lennard_jones(dist_ntnn,self.pots_ntnn), axis = 1)
        pots_ntnn_1d = self.Lennard_jones(dist_ntnn,self.pots_ntnn)
        pots_max_f_ntnn = self.Lennard_jones(dist_hist_ntnn,self.pots_ntnn)
        maxdist_ntnn = np.maximum(dist_ntnn,dist_hist_ntnn)
        pots_max_ntnn = self.softmax(pots_ntnn_1d,pots_max_f_ntnn)

        pots_der_ntnn = self.Lennard_jones_der(dist_ntnn,self.pots_ntnn)
        pots_der_ntnn_max = self.Lennard_jones_der(dist_hist_ntnn,self.pots_ntnn)

        pots_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))
        pots_max_f_ntnn_alpha = np.zeros((self.num_edges_ntnn,self.num_alpha))

        count = 0
        for entry in self.alpha_inds_ntnn:
            
            pots_ntnn_alpha[entry,count] = pots_ntnn_1d[entry]
            pots_max_f_ntnn_alpha[entry,count] = pots_max_f_ntnn[entry]
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
    
    def plot_eigs(self): ### deprecated
        fig, ax = plt.subplots(nrows=1)
        ind = np.where(self.eigs < self.min_eig)
        mine = np.argmin(self.eigs)
        ax.scatter(self.T, self.eigs)
        ax.scatter(self.T[ind], self.eigs[ind], color = 'r')
        ax.scatter(self.T[mine], self.eigs[mine], color = 'g',label = 'Min EV = %f'%(min(self.eigs)))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        fig.suptitle('Eigenvalue evolution, \n time needed: %f' %(self.time_needed))
        fig.legend()
        fig.savefig(self.folder + "/eig_ev.png" , bbox_inches="tight", dpi = 300)
        plt.close(fig)
        fig.clf()

    def plot_eigsn(self, print_type = 'png', suffix = '', zieldirectory = ''):
        if len(zieldirectory) == 0:
            im_path = self.folder 
        else:
            im_path = zieldirectory 
        if not os.path.exists(im_path): 
            os.makedirs(im_path)
            
        minind = np.argmin(self.eigsn[0,:])
        for i in range(self.num_eigs):
            fig, ax = plt.subplots(nrows=1)
            mine = np.argmin(self.eigsn[i,:])
            ax.plot(self.T, np.zeros(self.T.shape), c = 'red')
            ax.scatter(self.T, self.eigsn[i,:])
            if not i == 0:
                ax.scatter(self.T[minind], self.eigsn[i,minind], color = 'red',label = r'Time of Fracture $t_c = %f$ '%(self.T[minind]))
                ax.scatter(self.T[minind], self.eigsn[i,minind], color = 'red',label = 'Value at ToF: %f'%(self.eigsn[i,minind]))
                ax.scatter(self.T[mine], self.eigsn[i,mine], color = 'orange',label = 'Minimum Value: %f'%(min(self.eigsn[i,:])))
            else:
                ax.scatter(self.T[minind], self.eigsn[i,minind], color = 'red',label = 'Time of Fracture $t_c = %f$'%(self.T[minind]))
                ax.scatter(self.T[mine], self.eigsn[i,mine], color = 'red',label = 'Minimum Value: %f'%(min(self.eigsn[i,:])))
            
            print(i,'relevant eigval:', self.eigsn[i,minind])
            ax.set_ylabel(r'$\Lambda_%i$' %i, rotation = 0,fontsize = 12, labelpad = 10)
            ax.set_xlabel(r't',fontsize = 12)
            ax.legend(loc = 'upper right',fontsize = 12)
            if print_type == 'png':
                fig.savefig(im_path+ '/eig_ev_' + str(i) + suffix + ".png" , bbox_inches="tight", dpi = 300)
            else:
                fig.savefig(im_path+ '/eig_ev_' + str(i) + suffix + ".pdf" , bbox_inches="tight", dpi = 300)
            plt.close(fig)
            fig.clf()
            
    
    def plot_system(self,timelist, print_int, print_ntnn = False, print_type = 'png', suffix = '', zieldirectory = ''):

        if len(zieldirectory) == 0:
            im_path = self.folder + '/images'
        else:
            im_path = zieldirectory + '/images'
        if not os.path.exists(im_path): 
            os.makedirs(im_path)
        data_arr=np.linspace(0,1,1000)
        color=mcp.gen_color_normalized(cmap="coolwarm",data_arr=data_arr)

        if self.compute_mat_der:
            color_sens=mcp.gen_color_normalized(cmap="coolwarm",data_arr=data_arr)
            if len(zieldirectory) == 0:
                im_path_sens = im_path + '/sensitivities'
            else:
                im_path_sens = zieldirectory + '/sensitivities'
            if not os.path.exists(im_path_sens): 
                os.makedirs(im_path_sens)

        images = []
        images_ntnn = []
        print('################## started plotting ####################')
        print(timelist, print_int)
        for time in timelist:
            if time%print_int == 0:
                print('#-#-# Doing plot number %i now #-#-#' %(time))
            fig, ax = plt.subplots(nrows=1)
            verts = np.reshape(self.Y[:,time], (self.L*self.B,2))
            edge_hist, edge_index= self.get_edges(self.X,ret_index = True)        
            self.edge_index = np.where(edge_index <2, edge_index, -1).copy() ### prime the edge lists
            self.edge0_sort = np.where(self.edge_index.T == 1)
            self.edge1_sort = np.where(self.edge_index.T == -1)
            if self.ntnn:
                edge_hist_ntnn, edge_index_ntnn = self.get_edges_ntnn(self.X,ret_index = True)
                self.edge_index_ntnn = np.where(edge_index_ntnn <2, edge_index_ntnn, -1).copy()
                self.edge0_sort_ntnn = np.where(self.edge_index_ntnn.T == 1)
                self.edge1_sort_ntnn = np.where(self.edge_index_ntnn.T == -1)
            edges = np.zeros((self.num_edges,2,2))
            edges[self.edge0_sort[0],0,0] = self.Y[::2,time][self.edge0_sort[1]]
            edges[self.edge0_sort[0],0,1] = self.Y[1::2,time][self.edge0_sort[1]]
            edges[self.edge1_sort[0],1,0] = self.Y[::2,time][self.edge1_sort[1]]
            edges[self.edge1_sort[0],1,1] = self.Y[1::2,time][self.edge1_sort[1]]
            pots = self.Lennard_jones(self.hist_list[time],self.pots)
            damages = self.damage_var(self.hist_list[time])                
            remove=np.where(damages >= 1-0.001)
            edges_intact = np.delete(edges,remove,0)
            damages = np.delete(damages,remove,0)
            ax.scatter(verts[:,0],verts[:,1], zorder = 1)


            #### just so we get the indeces for the edges
            self.apply_material_parameters(edge_hist)

            
            for i,entry in enumerate(edges_intact):
                ax.plot(entry[:,0], entry[:,1], color = color[np.abs(data_arr - damages[i]).argmin()], zorder = 0)

            for entry in edges[self.alpha_inds[0],:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 2, c = 'red')
            for entry in edges[self.alpha_inds[1],:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 2, c = 'cyan')
            for ind_3 in self.alpha_fix_inds[1:]:
                for entry in edges[ind_3,:,:]:
                    ax.plot(entry[:,0], entry[:,1], zorder = 2, c = 'orange')
            for entry in edges[self.alpha_fix_inds[0],:,:]:
                ax.plot(entry[:,0], entry[:,1], zorder = 2, c = 'yellow')

            nodes = self.get_dirichlet_nodes()
            for entry in nodes: 
                dirichlet_nodes = np.reshape(self.Y[entry,time],(int(len(entry)/2),2)).copy()
                ax.scatter(dirichlet_nodes[:,0],dirichlet_nodes[:,1], zorder = 2)


            if self.ntnn and print_ntnn:
                fig_ntnn, ax_ntnn = plt.subplots(nrows=1)
                ax_ntnn.scatter(verts[:,0],verts[:,1], zorder = 1)
                edges_ntnn = np.zeros((self.num_edges_ntnn,2,2))
                edges_ntnn[self.edge0_sort_ntnn[0],0,0] = self.Y[::2,time][self.edge0_sort_ntnn[1]]
                edges_ntnn[self.edge0_sort_ntnn[0],0,1] = self.Y[1::2,time][self.edge0_sort_ntnn[1]]
                edges_ntnn[self.edge1_sort_ntnn[0],1,0] = self.Y[::2,time][self.edge1_sort_ntnn[1]]
                edges_ntnn[self.edge1_sort_ntnn[0],1,1] = self.Y[1::2,time][self.edge1_sort_ntnn[1]]
                pots_ntnn = self.Lennard_jones(self.hist_ntnn_list[time],self.pots_ntnn)
                damages_ntnn = self.damage_var_ntnn(self.hist_ntnn_list[time])
                remove_ntnn=np.where(damages_ntnn >= 1-0.001)
                edges_intact_ntnn = np.delete(edges_ntnn,remove_ntnn,0)
                damages_ntnn = np.delete(damages_ntnn,remove_ntnn,0)

                #### just so we get the indeces for the edges
                self.apply_material_parameters_ntnn(edge_hist_ntnn)

                for i,entry in enumerate(edges_intact_ntnn):
                    ax_ntnn.plot(entry[:,0], entry[:,1], color = color[np.abs(data_arr - damages_ntnn[i]).argmin()], zorder = 0)

                for entry in edges_ntnn[self.alpha_inds_ntnn[0],:,:]:
                    ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 2, c= 'red')
                for entry in edges_ntnn[self.alpha_inds_ntnn[1],:,:]:
                    ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 2, c = 'cyan')
                for ind_3 in self.alpha_fix_inds_ntnn[1:]:
                    for entry in edges_ntnn[ind_3,:,:]:
                        ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 2, c = 'orange')
                for entry in edges_ntnn[self.alpha_fix_inds_ntnn[0],:,:]:
                    ax_ntnn.plot(entry[:,0], entry[:,1], zorder = 2, c = 'yellow')

                for entry in nodes:
                    dirichlet_nodes = np.reshape(self.Y[entry,time],(int(len(entry)/2),2)).copy()
                    ax_ntnn.scatter(dirichlet_nodes[:,0],dirichlet_nodes[:,1], zorder = 2)#

            if self.compute_mat_der:
                verts_alpha = np.reshape(self.Y_alpha[:,:,time], (self.L*self.B,2,self.num_alpha))
                damage_sens = self.damage_var_mat(self.hist_list[time],self.hist_list_alpha[time])
                if self.ntnn: 
                    damage_sens_ntnn = self.damage_var_ntnn_mat(self.hist_ntnn_list[time],self.hist_ntnn_list_alpha[time])
                vmin = 0
                vmax= 0
                for k in range(self.num_alpha):
                    fig_mat, ax_mat = plt.subplots(nrows=2, sharex = True)
                    for ii, ax in enumerate(ax_mat.flat):
                        
                        in0 = ax_mat[ii].scatter(verts[:,0],verts[:,1], c = verts_alpha[:,ii,k])
                    if self.ntnn and print_ntnn:
                        for i,entry in enumerate(edges_ntnn):
                            ax_mat[0].plot(entry[:,0], entry[:,1], c= color[np.floor(damage_sens_ntnn[i,k]).astype(int)], zorder = 0)
                            ax_mat[1].plot(entry[:,0], entry[:,1], c= color[np.floor(damage_sens_ntnn[i,k]).astype(int)], zorder = 0)
                    else: 
                        for i,entry in enumerate(edges):
                            ax_mat[0].plot(entry[:,0], entry[:,1], c= color[np.floor(damage_sens[i,k]).astype(int)], zorder = 0)
                            ax_mat[1].plot(entry[:,0], entry[:,1], c= color[np.floor(damage_sens[i,k]).astype(int)], zorder = 0)


                    cmap = mpl.cm.viridis
                    norm = mpl.colors.Normalize(vmin=np.min(verts_alpha[:,:,:]), vmax=np.max(verts_alpha[:,:,:]))
                    sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    cbar = fig_mat.colorbar(sm,ax = ax_mat)
                    fig_mat.suptitle('Derivative with regard to alpha_%i '%k +'at t = %f'%self.T[time])
                    ax_mat[0].set_ylabel('x displacement')
                    ax_mat[1].set_ylabel('y displacement')
                    if print_type == 'png':
                        fig_mat.savefig(im_path_sens + '/der_alpha_%i'%(k) +'_timestep_%i'%(time) + suffix + '.png', bbox_inches="tight", dpi = 300)
                    else:
                        fig_mat.savefig(im_path_sens + '/der_alpha_%i'%(k) +'_timestep_%i'%(time) + suffix + '.pdf', bbox_inches="tight", dpi = 300)
                    plt.close(fig_mat)
                    fig_mat.clf()
                
                    
            fig.suptitle('Sytem state at t = %f'%self.T[time])
            if print_type == 'png':
                fig.savefig(im_path + "/timestep_%i"%(time) + suffix + ".png", bbox_inches="tight", dpi = 300)
                images.append(im_path + "/timestep_%i"%(time) + suffix + ".png" )
            else:
                fig.savefig(im_path + "/timestep_%i"%(time) + suffix + ".pdf", bbox_inches="tight", dpi = 300)
                images.append(im_path + "/timestep_%i"%(time) + suffix + ".pdf" )
            plt.close(fig)
            fig.clf()
            plt.pause(0.001)
            if self.ntnn and print_ntnn:
                fig_ntnn.suptitle('Sytem state at t = %f'%self.T[time])
                if print_type == 'png':
                    fig_ntnn.savefig(im_path + "/ntnn_timestep_%i"%(time) + suffix + ".png" , bbox_inches="tight", dpi = 300)
                    images_ntnn.append(im_path + "/ntnn_timestep_%i"%(time) + suffix + ".png")
                else:
                    fig_ntnn.savefig(im_path + "/ntnn_timestep_%i"%(time) + suffix + ".pdf", bbox_inches="tight", dpi = 300)
                    images_ntnn.append(im_path + "/ntnn_timestep_%i"%(time) + suffix + ".pdf" )
                plt.close(fig_ntnn)
                fig_ntnn.clf()
                
            plt.close('all')

        ### lists which keep track of where the images are and in what order
        self.images = images
        if self.ntnn: 
            self.images_ntnn = images_ntnn
        
        print('------- Done with generating plots -------')
        print('------- Find them in' + im_path + ' -------')
        print('################## finished plotting ####################')

        
    def create_anim(self,fps):
        ### creates animated video of the simulated breakage
        ### requires you to run plot_system first
        print('################## started animating ####################')
        im_path = self.folder + '/images'
        name = 'video'
        frame = cv2.imread(self.images[0])
        height, width, layers = frame.shape
        video = cv2.VideoWriter(self.folder + '/' + name + '.avi', 0, fps, (width,height))
        if self.ntnn:
            video_ntnn = cv2.VideoWriter(self.folder + '/' + name + '_ntnn' + '.avi', 0, fps, (width,height))
        for i,image in enumerate(self.images):
            video.write(cv2.resize(cv2.imread(image), (width,height)))
        if self.ntnn:
            for i,image in enumerate(self.images_ntnn):
                video_ntnn.write(cv2.resize(cv2.imread(image), (width,height)))
        cv2.destroyAllWindows()
        video.release()
        if self.ntnn:
            video_ntnn.release()
        print('################## finished animating ####################')
        

                


            
        
