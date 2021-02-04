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
import pickle as pickle
import sys

from torch.distributions import Independent, Distribution
sys.path.insert(1, '/home/mvasist/scripts/')
from fab_priors import BoxUniform_New

import torch
from sbi.inference import SNRE_A, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi.types import Array, OneOrMore, ScalarFloat

import emcee
from emcee.utils import MPIPool

#from Simulator import Simulator

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
# R_pl =   1.84*nc.r_jup_mean
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
Im using only data_flux_nu_error['MIRI LRS'] from here to calculate the likelihood. 
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

#  Monitor the range of the sampled prior

def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.

def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.

log_priors = {}

log_priors['t_int']          = lambda x: a_b_range(x, 0., 1500.)
log_priors['t_equ']          = lambda x: a_b_range(x, 0., 4000.)
log_priors['log_g']          = lambda x: a_b_range(x, 2.0, 3.7)

    
def Simulator_paul(params): 
    ##################

    NaN_spectra = 0

    atmosphere = Radtrans(line_species = ['H2O', 'CO_all_iso', \
                                         'CO2', 'CH4', \
                                          'Na', 'K'], \
              rayleigh_species = ['H2', 'He'], \
              continuum_opacities = ['H2-H2', 'H2-He'], \
              wlen_bords_micron = [0.3, 15])#, mode='c-k')

    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)
    temperature = 1200. * np.ones_like(pressures)

    R_pl = 1.838*nc.r_jup_mean
    
    log_g = 2.45                                #params[5]
    log_P0 = -2                                 #params[6] 

    kappa_IR = 0.01
    log_gamma = params[0]                            # log(0.4)
    T_int = params[1]                                 #200.
    T_equ = params[2]                                 #1500.
    
    gravity = np.exp(log_g)
    P0 = np.exp(log_P0)
    gamma = np.exp(log_gamma)

    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
    
    # Make dictionary for log 'metal' abundances    
    abundances = {}
    abundances['H2'] = 0.74 * np.ones_like(temperature)         #np.exp(params[3]) * np.ones_like(temperature)
    abundances['He'] = 0.24 * np.ones_like(temperature)         #np.exp(params[4]) * np.ones_like(temperature)
    abundances['H2O'] = 0.001 * np.ones_like(temperature)       #np.exp(params[7]) * np.ones_like(temperature)
    abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature) #np.exp(params[8]) * np.ones_like(temperature)
    abundances['CO2'] = 0.00001 * np.ones_like(temperature)     #np.exp(params[9]) * np.ones_like(temperature)
    abundances['CH4'] = 0.000001 * np.ones_like(temperature)    #np.exp(params[10]) * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)      #np.exp(params[11]) * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)      #np.exp(params[12]) * np.ones_like(temperature)
    
    
    # Make dictionary for modified Guillot parameters
    temp_params = {}
    temp_params['t_int'] = T_int
    temp_params['t_equ'] = T_equ
    
    # Prior calculation of all input parameters
    log_prior = 0.

    for key in temp_params.keys():
        log_prior += log_priors[key](temp_params[key])

    # Return -inf if parameters fall outside prior distribution
    if (log_prior == -np.inf):
        return -np.inf

    # Calculate the log-likelihood
    log_likelihood = 0.
    
    MMW = rm.calc_MMW(abundances) * np.ones_like(temperature)
    
    atmosphere.calc_flux(temperature, abundances, gravity, MMW)

    wlen, flux_nu = nc.c/atmosphere.freq/1e-4/10000, atmosphere.flux/1e-6
    

    # Just to make sure that a long chain does not die
    # unexpectedly:
    # Return -inf if forward model returns NaN values
    if np.sum(np.isnan(flux_nu)) > 0:
        print("NaN spectrum encountered")
        NaN_spectra += 1
        return -np.inf    #np.ones((1,371))*

    # Convert to observation for emission case
    flux_star = fstar(wlen)
    flux_sq   = flux_nu/flux_star*(R_pl/R_star)**2 
    
    # Rebin model to observation
    flux_rebinned = rgw.rebin_give_width(wlen, flux_sq, \
                    data_wlen['MIRI LRS'], data_wlen_bins['MIRI LRS'])

    ################################################additions################################################
    
    observation = np.array(torch.load('/home/mvasist/scripts/3_param_observation.pt').numpy())

    ####################################################################
    ####### Calculate log-likelihood
    ####################################################################
    
    log_likelihood = -np.sum(((flux_rebinned - observation)/ \
                       data_flux_nu_error['MIRI LRS'])**2.)/2.
                
    
    if np.isnan(log_prior + log_likelihood):
        return -np.inf
    else:
        return log_prior + log_likelihood

def lnprob(x):
    return Simulator_paul(x)

##############################################################################################################################
    
# Retrieval hyperparameters
stepsize = 1.75
n_walkers = 240
n_iter = 50

n_dim = len(log_priors)


p0 = [np.array([np.random.normal(loc = 0., scale = 2., size=1)[0], \
                0.+1500.*np.random.uniform(size=1)[0], \
                0.+4000.*np.random.uniform(size=1)[0],] \
                ) for i in range(n_walkers)]

##############################################################################################################################

# Multiprocessing

cluster = False         # Submit to cluster
n_threads = 30         # Use mutliprocessing (local = 1)
write_threshold = 200 # number of iterations after which diagnostics are updated


if cluster:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                        a = stepsize, pool = pool)
else:
    if n_threads > 1:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                            a = stepsize, threads = n_threads)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                            a = stepsize)
        
##############################################################################################################################

#  Pre-burn in chain
    
pre_burn_in_runs = int(np.min([399, n_iter/10])) + 3
pos, prob, state = sampler.run_mcmc(p0, pre_burn_in_runs)    
    
highest_prob_index = np.unravel_index(sampler.lnprobability.argmax(), \
                                          sampler.lnprobability.shape)
best_position = sampler.chain[highest_prob_index]

f = open('/home/mvasist/samples_paul_MCMC/best_position_pre_burn_in_' + retrieval_name + str(sys.argv[1]) + '.dat', 'w')
f.write(str(best_position))
f.close()

print('best pos is done')

##############################################################################################################################

# Run actual chain

p0 = [np.array([best_position[0]+np.random.normal(size=1)[0]*0.5, \
                best_position[1]+np.random.normal(size=1)[0]*70., \
                best_position[2]+np.random.normal(size=1)[0]*200.] \
                   ) for i in range(n_walkers)] 
    
if cluster:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                        a = stepsize, pool = pool)
else:
    if n_threads > 1:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                            a = stepsize, threads = n_threads)
    else:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, \
                                            a = stepsize)

pos, prob, state = sampler.run_mcmc(p0, n_iter)

if cluster:
    pool.close()
    
print('actual chain is done')

##############################################################################################################################
    
# Saving results   

f = open('/home/mvasist/samples_paul_MCMC/chain_pos_' + retrieval_name + str(sys.argv[1]) + '.pickle','wb')
pickle.dump(pos,f)
pickle.dump(prob,f)
pickle.dump(state,f)
samples = sampler.chain[:, :, :].reshape((-1, n_dim))
pickle.dump(samples,f)
f.close()


with open('/home/mvasist/samples_paul_MCMC/chain_lnprob_' + retrieval_name + str(sys.argv[1]) + '.pickle', 'wb') as f:
    pickle.dump([sampler.lnprobability], \
                f, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Saving results is done')
