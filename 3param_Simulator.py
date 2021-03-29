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
import sys

from torch.distributions import Independent, Distribution
sys.path.insert(1, '/home/mvasist/scripts/')
from fab_priors import BoxUniform_New

import torch
from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.types import Array, OneOrMore, ScalarFloat

op= '/home/mvasist/petitRADTRANS/petitRADTRANS/retrieval_examples/emission/'
observation_files = {}
# observation_files['NIRISS SOSS'] = op +'NIRISS_SOSS_flux.dat'
# observation_files['NIRSpec G395M'] = op +'NIRSpec_G395M_flux.dat'
observation_files['MIRI LRS'] = op +'MIRI_LRS_flux.dat'

# Wavelength range of observations, fixed parameters that will not be retrieved
WLENGTH = [0.3, 15.0]
# LOG_G =  2.58
R_pl =   1.84*nc.r_jup_mean
R_star = 1.81*nc.r_sun
gamma = 1
t_equ= 0
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

data_wlen = {}
data_flux_nu = {}
data_flux_nu_error = {}
data_wlen_bins = {}

for name in observation_files.keys():
#         print(name)
    dat_obs = np.genfromtxt(observation_files[name])
    data_wlen[name] = dat_obs[:,0]*1e-4
    data_flux_nu[name] = dat_obs[:,1]
    data_flux_nu_error[name] = dat_obs[:,2]

    data_wlen_bins[name] = np.zeros_like(data_wlen[name])
    data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
    data_wlen_bins[name][-1] = data_wlen_bins[name][-2]
        
def Simulator(params): 

    NaN_spectra = 0

    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', \
                                         'CO2', 'CH4', \
                                          'Na', 'K'], \
          rayleigh_species = ['H2', 'He'], \
          continuum_opacities = ['H2-H2', 'H2-He'], \
          wlen_bords_micron = WLENGTH)#, mode='c-k')


    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)
    temperature = 1200. * np.ones_like(pressures)

    
    t_int = params[0].numpy()                             #200.
    log_kappa_IR = params[1].numpy()                      #-2
    log_gravity = params[2].numpy()                       #params[5].numpy() 1e1**2.45 

    gravity = np.exp(log_gravity)
    kappa_IR = np.exp(log_kappa_IR)
    
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, t_int, t_equ)
    
    abundances = {}
    abundances['H2'] = 0.75 * np.ones_like(temperature) #0.74 * np.ones_like(temperature) (params[3].numpy())
    abundances['He'] = 0.25 * np.ones_like(temperature)  #0.24 * np.ones_like(temperature) (params[4].numpy())
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
    abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
    abundances['CO2'] = 0.00001 * np.ones_like(temperature)
    abundances['CH4'] = 0.000001 * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)

    MMW = rm.calc_MMW(abundances) * np.ones_like(temperature)
    #print(MMW, abundances)
    
    atmosphere.calc_flux(temperature, abundances, gravity, MMW)

    wlen, flux_nu = nc.c/atmosphere.freq, atmosphere.flux/1e-6


    # Just to make sure that a long chain does not die
    # unexpectedly:
    # Return -inf if forward model returns NaN values
    if np.sum(np.isnan(flux_nu)) > 0:
        print("NaN spectrum encountered")
        NaN_spectra += 1
        return torch.ones([1,371])* -np.inf

    # Convert to observation for emission case
    flux_star = fstar(wlen)
    flux_sq   = flux_nu/flux_star*(R_pl/R_star)**2 

    flux_rebinned = rgw.rebin_give_width(wlen, flux_sq, \
                data_wlen['MIRI LRS'], data_wlen_bins['MIRI LRS'])

    #flux_rebinned = np.reshape(flux_rebinned, (371,1))    

    FR= torch.Tensor(flux_rebinned)
    
    return FR    

# Prior= BoxUniform_New(a=torch.tensor([0.4, 0, 0, 0.64, 0.14, np.exp(2.45), 0.01]), \
#                              b=torch.tensor([0.1, 1500, 4000, 0.84, 0.34, np.exp(2.45), 0.01]))

Prior= utils.BoxUniform(low=torch.tensor([0., -4 , 2 ]), high=torch.tensor([2000., 0, 3.7 ]))
                             
sim = 1000

simulator, prior = prepare_for_sbi(Simulator, Prior)

inference = SNRE_A(prior= prior, device= 'cpu')

#records every 100 sim
for i in range(0, int(sim/100)):
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations= 100)
    T = pd.DataFrame(theta.numpy())
    X = pd.DataFrame(x.numpy())
    X.to_csv('/home/mvasist/simulations/3_params/1/X_100ksim_TintLkIRLg' + str(sys.argv[1]) + '.csv',mode='a', header=False)
    T.to_csv('/home/mvasist/simulations/3_params/1/T_100ksim_TintLkIRLg' + str(sys.argv[1]) + '.csv',mode='a', header=False)
