import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.tri as tri
import datetime
from matplotlib.colors import ListedColormap
from scipy.integrate import quad, dblquad

from Gradient_Descent_variable_circle import GradientDescent
import matplotlib.patches as patches

def Spring_fct(x,args = ()):
    eigs = []
    eigs_der = []
    vals = []
    derivs = []

    evalue = 5

    push = 0
    push2 = 0

    ### discontinuous parabola, with instant drop at x= 0

    for entry in x:

        if entry[0]-entry[1] +push >= 0 and entry[1]+entry[0] +push2>= 0:
            vals.append( np.abs(entry[1])+np.abs(entry[0]-5) -6*evalue
                    )
        elif entry[0]-entry[1] +push >= 0 and entry[1]+entry[0] +push2< 0:
            vals.append( np.abs(entry[0]+entry[1]) +2*evalue
                    )
        elif entry[0]-entry[1] +push < 0 and entry[1]+entry[0] +push2>= 0:
            vals.append( np.abs(entry[0]-entry[1]) +evalue
                    )
        elif entry[0]-entry[1] +push < 0 and entry[1]+entry[0] +push2< 0:
            vals.append( np.abs(entry[0]) + np.abs(entry[1]) +np.abs(entry[1]+entry[0]) + 2*evalue
                    )
        if len(args)>=1:
            df_x = np.zeros(2)  
            if entry[0]-entry[1] +push> 0 and entry[1]+entry[0] +push2> 0:
                df_x[1] = np.sign(entry[1])
                df_x[0] = np.sign(entry[0])
            elif entry[0]-entry[1] +push> 0 and entry[1]+entry[0] +push2< 0:
                df_x[1] = np.sign(entry[1]+entry[0])
                df_x[0] = np.sign(entry[1]+entry[0])
            elif entry[0]-entry[1] +push< 0 and entry[1]+entry[0] +push2> 0:
                df_x[1] = -np.sign(-entry[1]+entry[0])
                df_x[0] = np.sign(-entry[1]+entry[0])
            elif entry[0]-entry[1] +push<= 0 and entry[1]+entry[0] +push2 <= 0:
                df_x[0] = np.sign(entry[0])+np.sign(entry[1]+entry[0])
                df_x[1] = np.sign(entry[1])+np.sign(entry[1]+entry[0])
            derivs.append(df_x
                )
        if len(args)>=2:
            eigs.append([np.abs(entry[0]-entry[1]) +push,np.abs(entry[0]+entry[1]) +push2])
            eigs_der.append([np.asarray([1,-1]), np.asarray([1,1])])
    if len(args)>=2:
        return np.asarray(vals), np.asarray(derivs),np.asarray(eigs), np.asarray(eigs_der)
    elif len(args)>=1:
        return np.asarray(vals), np.asarray(derivs)
    else: 
        return np.asarray(vals)



date_time = 'final_circ_ex_new'
pathing = './Ex_nGs_' + date_time ### path to dataset
metadata = pathing + '/metadata.npy'

### load metadata
meta = np.load(metadata)
max_try = meta['max_try']
iter_max = meta['iter_max']
multi = meta['multi']

for i in range(0,max_try):
    filename1 = pathing + '/Run_' + str(i) + '_results1.npy'
    filename3 = pathing + '/Run_' + str(i) + '_results3.npy'
        
    f3 = np.load(filename3)
    sol3 = f3['sol3']
    
    
    f1 = np.load(filename1)
    sol1 = f1['sol1']

  
fmin = Spring_fct
fmincon = lambda x: Spring_fct(x,['Derivative'])

pop_per_dir = 10
sigma = 1
eps_val = [np.sqrt(9)]
kappa = [np.sqrt(4)]
aleph = 5

resolution = 100

xx = np.linspace(-20,20,resolution)
xx1,xx2 = np.meshgrid(xx,xx)
zz = np.ones((resolution*resolution,1))
zs = np.ones((resolution*resolution,1))
zst = np.ones((resolution*resolution,1))
zsingle = np.ones((resolution*resolution,1))
zder1 = np.ones((resolution*resolution,1))
zder2 = np.ones((resolution*resolution,1))

optst = GradientDescent(
        xstart=[0, 0], npop= pop_per_dir**2, sigma=sigma, gradient_available = False, 
        npop_der = 1,npop_stoch = pop_per_dir, num_id_funcs = 0
    )
    

    ### start optimization 
optst.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)

opt = GradientDescent(
        xstart=[0, 0], npop=1, sigma=sigma, gradient_available = True, 
        check_for_stability = True, use_one_directional_smoothing = True,
        npop_der = 1,npop_stoch = pop_per_dir, num_id_funcs = 2
    )
    

    ### start optimization 
opt.optimize(fmin, iterations=1, gradient_function = None, eps_val = eps_val, kappa = kappa, aleph = aleph)

ii = 0
opt.gradient_available = False
min_coords_ad = sol3[-1]
min_coords_g = sol1[-1]
minval_ad = 1000
minval_g = 1000
xstart = [0,15]
for entry in np.c_[xx1.ravel(), xx2.ravel()]:
    (zz[ii],dd,gg,dgg) = Spring_fct([entry], [9,3])
    zs[ii] = opt.compute_val(entry)
    zder1[ii] = np.squeeze(opt.compute_grad(entry))[0]
    zder2[ii] = np.squeeze(opt.compute_grad(entry))[1]
    zst[ii] = optst.compute_val(entry)
    ii+=1
opt.gradient_available = False
plt.show()
zz = zz.reshape(xx1.shape)
zs = zs.reshape(xx1.shape)
zst = zst.reshape(xx1.shape)
zder1 = zder1.reshape(xx1.shape)
zder2 = zder2.reshape(xx1.shape)
zsingle = zsingle.reshape(xx1.shape)


fig1,ax1 = plt.subplots(nrows = 1)
### colour map for landscape plots
cividis_new = plt.colormaps['viridis']
cmap = ListedColormap(cividis_new(np.linspace(0, 1, 256)))
CS = ax1.contourf(xx1, xx2, zz, cmap = cmap, levels = 40, zorder = 1)
cbar = fig1.colorbar(CS)

x_c = np.linspace(-20,20,100)

ax1.plot(x_c, x_c + eps_val[0], c = 'orange', linestyle = '--')
ax1.plot(x_c, x_c - eps_val[0], c = 'orange', linestyle = '--')
ax1.plot(x_c, x_c + kappa[0], c = 'r', linestyle = '--')
ax1.plot(x_c, x_c - kappa[0], c = 'r', linestyle = '--')

ax1.plot(x_c, -x_c + eps_val[0], c = 'purple', linestyle = '--')
ax1.plot(x_c, -x_c - eps_val[0], c = 'purple', linestyle = '--')
ax1.plot(x_c, -x_c + kappa[0], c = 'magenta', linestyle = '--')
ax1.plot(x_c, -x_c - kappa[0], c = 'magenta', linestyle = '--')
ax1.set_ylim(-20,20)
ax1.set_xlim(-20,20)

xc = [0, 1.1, 1.25,    2.8, 3.71, 4.34, 3.57,1.5,    2.8, 3.71, 4.34, 3.57,1.5]
yc = [0, 1.25, -1.1,   3.75, 5.9, 7.12, 8.15,8.72,  -3.75, -5.9, -7.12, -8.15,-8.72]
dir1 = np.zeros((len(xc),2))
dir2 = np.zeros((len(xc),2))
len1 = np.zeros((len(xc),))
len2 = np.zeros((len(xc),))
num_both = 3
num_each = 5

for kk in range(len(xc)):
    (cc,dd,gg,dgg) = Spring_fct([[xc[kk], yc[kk]]], [9,3])
    dir1[kk,:] = dgg[0][0][:]
    dir2[kk,:] = dgg[0][1][:]
    if gg[0][0]**2 < eps_val[0]**2:
        z = ((gg[0][0])**2-kappa[0]**2)/(eps_val[0]**2 - kappa[0]**2)
        if z >=0: 
            len1[kk] =  ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3)
        else:
            len1[kk] = 1
    else:
        len1[kk] = 0
    if gg[0][1]**2 < eps_val[0]**2:
        z = ((gg[0][1])**2-kappa[0]**2)/(eps_val[0]**2 - kappa[0]**2)
        if z >=0: 
            len2[kk] =  ( 6*(1-z)**5 - 15*(1-z)**4 + 10*(1-z)**3) 
        else:
            len2[kk] = 1
    else:
        len2[kk] = 0


ax1.scatter(xc[num_both:num_both + num_each], yc[num_both:num_both + num_each], color = '#39FF14', s= 5 )
ax1.scatter(xc[num_both+ num_each:num_both + 2*num_each], yc[num_both+ num_each:num_both + 2*num_each], color = 'cyan', s= 5 )

for k in range(3,len(xc)):
    ax1.annotate("", xytext= (xc[k], yc[k]), xy=(xc[k] + len1[k]*dir1[k,0], yc[k] + len1[k]*dir1[k,1]), arrowprops=dict(arrowstyle="-",color = '#39FF14',linewidth=4))
    ax1.annotate("", xytext= (xc[k], yc[k]), xy=(xc[k] - len1[k]*dir1[k,0], yc[k] - len1[k]*dir1[k,1]), arrowprops=dict(arrowstyle="-",color = '#39FF14',linewidth=4))
    ax1.annotate("", xytext= (xc[k], yc[k]), xy=(xc[k] + len2[k]*dir2[k,0], yc[k] + len2[k]*dir2[k,1]), arrowprops=dict(arrowstyle="-",color = 'cyan',linewidth=4))
    ax1.annotate("", xytext= (xc[k], yc[k]), xy=(xc[k] - len2[k]*dir2[k,0], yc[k] - len2[k]*dir2[k,1]), arrowprops=dict(arrowstyle="-",color = 'cyan',linewidth=4))

for k in range(3):
    p0 = np.array([xc[k], yc[k]])
    cor0 = np.array([p0 + len2[k]*dir2[k,:] + len1[k]*dir1[k,:], p0 - len2[k]*dir2[k,:] + len1[k]*dir1[k,:],
              p0 - len2[k]*dir2[k,:] - len1[k]*dir1[k,:], p0 + len2[k]*dir2[k,:] - len1[k]*dir1[k,:]])
    rect0 = patches.Polygon(
        cor0, closed = True,  # bottom-left corner
        facecolor="#1DFF8A",
        zorder = 3,# transparency (0=transparent, 1=solid)
        alpha = 0.6
    )
    ax1.add_patch(rect0)
ax1.set_ylim(-8,8)
ax1.set_xlim(-8,8)

fig1.suptitle(r'Landscape of $f$')
ax1.set_xlabel(r'$\mathbf{x}[1]$')
ax1.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)


fig1a,ax1a = plt.subplots(nrows = 1)
cividis_new1a = plt.colormaps['viridis']
cmap1a = ListedColormap(cividis_new1a(np.linspace(0, 1, 256)))
CS1a = ax1a.contourf(xx1, xx2, zz, cmap = cmap, levels = 40, zorder = 1)
cbar1a = fig1a.colorbar(CS1a)
fig1a.suptitle(r'Landscape of $f$')
ax1a.set_xlabel(r'$\mathbf{x}[1]$')
ax1a.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)


fig2c,ax2c = plt.subplots(nrows = 1)
### colour map for landscape plots
cividis_new2c = plt.colormaps['viridis']
cmap2c = ListedColormap(cividis_new2c(np.linspace(0, 1, 256)))
CS2c = ax2c.contourf(xx1, xx2, zst, cmap = cmap, levels = 40, zorder = 1)
cbar2c = fig2c.colorbar(CS2c)
ax2c.scatter(sol1[:,0], sol1[:,1], c ='aqua', label = 'GSM', marker = 's')
ax2c.scatter(xstart[0],xstart[1],c = 'magenta', label = 'Starting Point')
ax2c.scatter(min_coords_g[0], min_coords_g[1],c = 'red', label = 'Minimum' )
ax2c.legend(loc = 'lower left')
fig2c.suptitle(r'Landscape of $F$')
ax2c.set_xlabel(r'$\mathbf{x}[1]$')
ax2c.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)

fig2,ax2 = plt.subplots(nrows = 1)
### colour map for landscape plots
cividis_new2 = plt.colormaps['viridis']
cmap2 = ListedColormap(cividis_new2(np.linspace(0, 1, 256)))
CS2 = ax2.contourf(xx1, xx2, zs, cmap = cmap, levels = 40, zorder = 1)
cbar2 = fig2.colorbar(CS2)
x_c = np.linspace(-20,20,100)

ax2.plot(x_c, x_c + eps_val[0], c = 'orange', linestyle = '--', zorder = 2)
ax2.plot(x_c, x_c - eps_val[0], c = 'orange', linestyle = '--', zorder = 2)
##ax2.plot(x_c, x_c + kappa[0]*np.sqrt(2), c = 'r', linestyle = '--') ### you can also plot Omega_G^kappa if you want to
##ax2.plot(x_c, x_c - kappa[0]*np.sqrt(2), c = 'r', linestyle = '--')
ax2.plot(x_c, -x_c + eps_val[0], c = 'purple', linestyle = '--', zorder = 2)
ax2.plot(x_c, -x_c - eps_val[0], c = 'purple', linestyle = '--', zorder = 2)
##ax2.plot(x_c, -x_c + kappa[0]*np.sqrt(2), c = 'magenta', linestyle = '--')
##ax2.plot(x_c, -x_c - kappa[0]*np.sqrt(2), c = 'magenta', linestyle = '--')
ind_g1 = np.where(np.absolute(sol3[:,0]-sol3[:,1])**2  < eps_val[0]**2)
ind_g2 = np.where(np.absolute(sol3[:,0]+sol3[:,1])**2  < eps_val[0]**2)
ind_gb = np.intersect1d(ind_g1, ind_g2)
ind_3 = np.union1d(ind_g1, ind_g2)
ind_0 = np.setdiff1d(range(len(sol3[:,0])), ind_3)
ind_3 = np.setdiff1d(ind_3, ind_gb)
ax2.scatter(sol3[ind_0,0], sol3[ind_0,1], c = 'yellow', label = 'ASM' ,  marker = 'D', zorder = 3)
ax2.scatter(xstart[0],xstart[1],c = 'magenta', label = 'Starting Point', zorder = 3)
ax2.scatter(sol3[ind_3,0], sol3[ind_3,1], c = 'darkorange', label = '1-dim Smoothing is applied', marker = '<', s = 70, zorder = 3)
ax2.scatter(sol3[ind_gb,0], sol3[ind_gb,1], c = 'magenta', label = '2-dim Smoothing is applied', marker = '<', s = 70, zorder = 3)
ax2.scatter(min_coords_ad[0], min_coords_ad[1],c = 'red', label = 'Minimum', zorder = 3)
ax2.set_ylim(-20,20)
ax2.set_xlim(-20,20)
ax2.set_xlabel(r'$\mathbf{x}[1]$')
ax2.set_ylabel(r'$\mathbf{x}[2]$', rotation = 0)
ax2.legend(loc = 'lower left')

fig2.suptitle(r'Landscape of $\mathbb{F}$')

plt.show()
    
