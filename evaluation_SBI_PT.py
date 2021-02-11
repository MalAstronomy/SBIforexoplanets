import time
from petitRADTRANS import Radtrans
import petitRADTRANS.retrieval_examples.emission.master_retrieval_model as rm
from petitRADTRANS import nat_cst as nc
import petitRADTRANS.rebin_give_width as rgw
from scipy.interpolate import interp1d
import sklearn

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
import pickle
import time

from torch.distributions import Independent, Distribution
from fab_priors import BoxUniform_New

import torch
from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.types import Array, OneOrMore, ScalarFloat

retrieval_name = 'JWST_emission_petitRADTRANSpaper'
absolute_path = 'output/'# end with forward slash!
op= '/home/mvasist/petitRADTRANS/petitRADTRANS/retrieval_examples/emission/'
observation_files = {}
observation_files['NIRISS SOSS'] = op +'NIRISS_SOSS_flux.dat'
observation_files['NIRSpec G395M'] = op +'NIRSpec_G395M_flux.dat'
observation_files['MIRI LRS'] = op +'MIRI_LRS_flux.dat'

# Wavelength range of observations, fixed parameters that will not be retrieved
WLEN = [0.3, 15.0]
# LOG_G =  2.58
R_pl = 1.838*nc.r_jup_mean
R_star = 1.81*nc.r_sun
# Get host star spectrum to calculate F_pl / F_star later.
T_star = 6295.
x = nc.get_PHOENIX_spec(T_star)
fstar = interp1d(x[:,0], x[:,1])

####################################################################################
####################################################################################
### READ IN OBSERVATION
####################################################################################
####################################################################################

# Read in data, convert all to cgs! 

'''
Im using only data_flux_nu_error['MIRI LRS'] from here (to calculate the likelihood-paul and rebinning flux-sbi). 
'''

data_wlen = {}
data_flux_nu = {}
data_flux_nu_error = {}
data_wlen_bins = {}

for name in observation_files.keys():
    print(name)
    dat_obs = np.genfromtxt(observation_files[name])
    data_wlen[name] = dat_obs[:,0]*1e-4
    data_flux_nu[name] = dat_obs[:,1]
    data_flux_nu_error[name] = dat_obs[:,2]
    
    data_wlen_bins[name] = np.zeros_like(data_wlen[name])
    data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
    data_wlen_bins[name][-1] = data_wlen_bins[name][-2]

################################################################################################################################

# Defining the atmosphere model 

atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', \
                                         'CO2', 'CH4', \
                                          'Na', 'K'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = [0.3, 15])#, mode='c-k')

gravity = np.exp(2.45)
pressures = np.logspace(-6, 2, 100)

atmosphere.setup_opa_structure(pressures)

################################################################################################################################

# Observational temperature profile 

kappa_IR = 0.01
gamma0 = 0.4
T_int0 = 750.
T_equ0 = 2000.

temperature0 = nc.guillot_global(pressures, kappa_IR, gamma0, gravity, T_int0, T_equ0)

################################################################################################################################

# Retrieving saved samples 
ss=[]

for i in ['100']:  #'12', '50', 
    s= pd.read_csv( '/home/mvasist/samples/' + i +'ksamples__SBI_100ksim.csv')    
    ss.append(s.values)

sss= np.vstack(ss)
df_samples =  pd.DataFrame(sss)

################################################################################################################################

# defining extremities

gamma1 = 0.01
T_int1 = 0.01
T_equ1 = 0.01

temperature1 = nc.guillot_global(pressures, kappa_IR, gamma1, gravity, T_int1, T_equ1)

gamma2 = 0.69
T_int2 = 1500
T_equ2 = 4000

temperature2 = nc.guillot_global(pressures, kappa_IR, gamma2, gravity, T_int2, T_equ2)

################################################################################################################################

# abundances and MMW

abundances = {}

abundances['H2'] = 0.74* np.ones_like(pressures) # params[3].numpy() * np.ones_like(temperature)
abundances['He'] = 0.24* np.ones_like(pressures)  # params[4].numpy() * np.ones_like(temperature)
abundances['H2O'] = 0.001 * np.ones_like(pressures)
abundances['CO_all_iso'] = 0.01 * np.ones_like(pressures)
abundances['CO2'] = 0.00001 * np.ones_like(pressures)
abundances['CH4'] = 0.000001 * np.ones_like(pressures)
abundances['Na'] = 0.00001 * np.ones_like(pressures)
abundances['K'] = 0.000001 * np.ones_like(pressures)

MMW = rm.calc_MMW(abundances) * np.ones_like(pressures)

################################################################################################################################

# picking 1000 random samples from the 100k samples generated and calculating their rebinned fluxes

temp = []
f=[]
w=[]
fr=[]

start = time.time()
for ind, i in enumerate(np.random.randint(0,99999,1000)):
    print(ind,i)
    index, log_gamma, T_int, T_equ = [df_samples[row][i] for row in df_samples]
    gamma = np.exp(log_gamma)
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ) 
    temp.append(temperature)
    atmosphere.calc_flux(temperature, abundances, gravity, MMW)
    wlen, flux_nu = nc.c/atmosphere.freq, atmosphere.flux/1e-6   
    
    flux_star = fstar(wlen)
    flux_sq   = flux_nu/flux_star*(R_pl/R_star)**2 
    
    flux_rebinned = rgw.rebin_give_width(wlen, flux_sq, \
                    data_wlen['MIRI LRS'], data_wlen_bins['MIRI LRS'])
            
        
    f.append(flux_nu)
    w.append(wlen)
    fr.append(flux_rebinned)
    
end= time.time()
print((end-start)/60, ' minutes')

    #wlen, flux_nu = nc.c/atmosphere.freq/1e-4/10000, atmosphere.flux/1e-6

#     plt.plot(nc.c/atmosphere.freq/1e-4, atmosphere.flux/1e-6)

#     plt.xscale('log')
#     plt.xlabel('Wavelength (microns)')
#     plt.ylabel(r'Planet flux $F_\nu$ (10$^{-6}$ erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)')
# plt.show()
# plt.clf()

################################################################################################################################

# A function that checks if any two or more spectra are identical out of the 1000 randomly chosen samples 

def similar(a): 
    a = np.stack(a, axis=0 )
    a=a.tolist()
    repeats ={}
    for h in range(len(a)):
        repeats[str(h)] = 1
    l=0
    e = []
    for i in range(len(a)):
        if (i not in e):
            for j in range(len(a)):
                if i==j : continue
                elif (a[i] == a[j]):
                    e.append(j)
                    repeats[str(i)] = repeats[str(i)] + 1
          
    d={}
    b = np.array(list(repeats.values())) 
    for k in np.where(b>1)[0]:
        d[str(k)] = repeats[str(k)]
        
    return d    

################################################################################################################################

# Function call 

similar(fr)

################################################################################################################################

# plotting PT profiles with lof prob

# creating P-T profile for each - takes around 10 min
temp = []
cc=[]
fig, ax = plt.subplots(1, figsize=(10,10))
plt.title('SBI 100ksamp e76 200kSim')
start = time.time()
c1=0
c2=0
c3=0
c4=0

for i in np.random.randint(0,199999, 10):  #np.arange(0,99999):  #
    index, log_gamma, T_int, T_equ = [df_samples[row][i] for row in df_samples]
    ind, lnprob = [df_lnprob[row][i] for row in df_lnprob]
    cc.append(color(lnprob)) 
    gamma = np.exp(log_gamma)
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    if ((color(lnprob)> 95) and (color(lnprob)<100)):
        ax.plot(temperature, pressures, c='darkblue', label = "95<prob<100" if c1 == 0 else "") #viridis,jet,gray,parula(?),magma,plasma,inferno-plt.cm.inferno(color(lnprob))
        c1+=1
    elif ((color(lnprob)> 80) and (color(lnprob)<95)):
        ax.plot(temperature, pressures, c='dodgerblue',label = "80<prob<95" if c2 == 0 else "") #if i == 0 else ""
        c2+=1
    else :
        ax.plot(temperature, pressures, c='skyblue', label = "prob<80" if c3 == 0 else "")
        c3+=1
    ax.plot(temperature0, pressures, c= 'red', label = 'observation value' if c4 == 0 else "")
    c4+=1
#     plt.plot(temperature1, pressures, c= 'red')
#     plt.plot(temperature2, pressures, c= 'red')
    plt.yscale('log')
    plt.ylim([1e2, 1e-6])
    plt.xlabel('T (K)')
    plt.ylabel('P (bar)')
    temp.append(temperature)
#plt.savefig('/home/mvasist/results/SNRE/PT_profile/SBI_162ksamp_100kSim.png')
handles,labels = ax.get_legend_handles_labels()
handles = [handles[0], handles[2],  handles[3], handles[1]]
labels = [labels[0], labels[2], labels[3], labels[1]]
ax.legend(handles,labels)
plt.show()

end =time.time()
print('it takes: '+ str((end-start)/60) + ' min')

