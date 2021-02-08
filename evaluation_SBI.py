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


retrieval_name = 'JWST_emission_petitRADTRANSpaper'
absolute_path = 'output/'# end with forward slash!
op= '/home/mvasist/petitRADTRANS/petitRADTRANS/retrieval_examples/emission/'
observation_files = {}
# observation_files['NIRISS SOSS'] = op +'NIRISS_SOSS_flux.dat'
# observation_files['NIRSpec G395M'] = op +'NIRSpec_G395M_flux.dat'
observation_files['MIRI LRS'] = op +'MIRI_LRS_flux.dat'

# Wavelength range of observations, fixed parameters that will not be retrieved
# WLEN = [0.8, 14.0]
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

data_wlen = {}
data_flux_nu = {}
data_flux_nu_error = {}
data_wlen_bins = {}

for name in observation_files.keys():
    dat_obs = np.genfromtxt(observation_files[name])
    data_wlen[name] = dat_obs[:,0]*1e-4
    data_flux_nu[name] = dat_obs[:,1]
    data_flux_nu_error[name] = dat_obs[:,2]

    data_wlen_bins[name] = np.zeros_like(data_wlen[name])
    data_wlen_bins[name][:-1] = np.diff(data_wlen[name])
    data_wlen_bins[name][-1] = data_wlen_bins[name][-2]
        
####################################################################################################################

def Simulator(params): 

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
    gravity = 1e1**2.45                           #1e1**2.45  params[5].numpy()
    P0 = 0.01                                     #0.01       params[6].numpy() 

    kappa_IR = 0.01
    log_gamma = params[0].numpy()                         # log 1.5 - 0.4
    T_int = params[1].numpy()                             #200.
    T_equ = params[2].numpy()                             #1500.
    
    gamma = np.exp(log_gamma)
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    abundances = {}
    abundances['H2'] = 0.74 * np.ones_like(temperature) #0.74 * np.ones_like(temperature) (params[3].numpy())
    abundances['He'] = 0.24 * np.ones_like(temperature)  #0.24 * np.ones_like(temperature) (params[4].numpy())
    abundances['H2O'] = 0.001 * np.ones_like(temperature)
    abundances['CO_all_iso'] = 0.01 * np.ones_like(temperature)
    abundances['CO2'] = 0.00001 * np.ones_like(temperature)
    abundances['CH4'] = 0.000001 * np.ones_like(temperature)
    abundances['Na'] = 0.00001 * np.ones_like(temperature)
    abundances['K'] = 0.000001 * np.ones_like(temperature)

    MMW = rm.calc_MMW(abundances) * np.ones_like(temperature)

    atmosphere.calc_flux(temperature, abundances, gravity, MMW)

    wlen, flux_nu = nc.c/atmosphere.freq/1e-4/10000, atmosphere.flux/1e-6


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

    FR= torch.Tensor(flux_rebinned)

    return FR

####################################################################################################################

Prior= BoxUniform_New(a=torch.tensor([0., 0, 0 ]), b=torch.tensor([2., 1500, 4000]))
                             
sim = 10000

simulator, prior = prepare_for_sbi(Simulator, Prior)

inference = SNRE_A(prior= prior, device= 'cpu')

####################################################################################################################

#reading the csv file

X=[]
T=[]
for i in range(1,11):
#     print(i)
    dfX= pd.read_csv('/home/mvasist/simulations/3_params/X_'+ str(i) + '.csv')
    dfT= pd.read_csv('/home/mvasist/simulations/3_params/T_'+ str(i) + '.csv')
    X.append(dfX.values)
    T.append(dfT.values)
    
comb_np_array_X = np.vstack(X)
comb_np_array_T = np.vstack(T)

Xframe = pd.DataFrame(comb_np_array_X)
Tframe = pd.DataFrame(comb_np_array_T)

list_of_tensors_X = [torch.tensor(np.array(Xframe),dtype=torch.float32)]
list_of_tensors_T = [torch.tensor(np.array(Tframe),dtype=torch.float32)]
XX = torch.cat(list_of_tensors_X)[:, 1:]
TT = torch.cat(list_of_tensors_T)[:,1:]
####################################################################################################################

inference = inference.append_simulations(TT, XX)

density_estimator = inference.train()

posterior = inference.build_posterior(density_estimator)

####################################################################################################################

observation = torch.load('3_param_observation.pt') #log_gamma, T_int, T_equ = 1.5, 750, 2000

####################################################################################################################
# Generating samples

sampls= 100000
samples = posterior.sample((sampls,), x=observation)
log_probability = posterior.log_prob(samples, x= observation)

####################################################################################################################
# Saved samples 

ss=[]

for i in ['100']: #, '50', '12'
    s= pd.read_csv( i +'ksamples__SBI_100ksim.csv')   
    ss.append(s.values)

sss= np.vstack(ss)
samples =  pd.DataFrame(sss)
samples = torch.tensor(np.array(samples),dtype=torch.float32)
samples[:, 1:].size()
samples = samples[:, 1:]

####################################################################################################################
points = torch.cat([torch.ones(1)*1.5,torch.ones(1)*750.,torch.ones(1)*2000.])

fig, axes = utils.pairplot(samples, points, limits=[[0.,2.],[0.,2000],[1500.,2500]], fig_size=(7,7), \
                           labels = ['log_gamma \n normal(0.,2.)', 'Tint \n U(0.,1500)', 'Tequ \n U(0.,4000)'],\
                           title= 'SNRE \n BU_new prior :' + str(100) + 'k simulations, '+ str(100) + 'k samples',\
                          points_colors = ['red'])

####################################################################################################################
