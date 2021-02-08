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

# Retrieving the samples
 
fname2 ='/home/mvasist/samples_paul_MCMC/chain_pos_JWST_emission_petitRADTRANSpaper30.pickle'
with open(fname2, 'rb') as f:
    lines2 = pickle.load(f)      #pos
    l3= pickle.load(f)           #prob
    l4= pickle.load(f)           #state
    l5= pickle.load(f)           #samples

sampls = l5

################################################################################################################################

## creating P-T profile for each - takes around 15 min
tem = []
for i in range(12000):   
    gamma = np.exp(sampls[i, 0])
    T_int = sampls[i, 1]
    T_equ = sampls[i, 2]
    temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
#     plt.plot(temperature, pressures, c= 'grey')
#     plt.plot(temperature0, pressures, c= 'black')
#     plt.yscale('log')
#     plt.ylim([1e2, 1e-6])
#     plt.xlabel('T (K)')
#     plt.ylabel('P (bar)')
    tem.append(temperature)
plt.show()
# plt.clf()

################################################################################################################################

# Checking how many samples are 0s

len(np.where(np.sum(tem, axis= 1) == 0)[0])  

