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
import pickle as pickle
import corner

#################################################################################################################

fname ='/home/mvasist/samples_paul_MCMC/best_position_pre_burn_in_JWST_emission_petitRADTRANSpaper30.dat'
with open(fname, 'rb') as f:
    lines = [x.decode('utf8').strip() for x in f.readlines()]

fname1 ='/home/mvasist/samples_paul_MCMC/chain_lnprob_JWST_emission_petitRADTRANSpaper30.pickle'
with open(fname1, 'rb') as f:
    lines11 = pickle.load(f)

fname2 ='/home/mvasist/samples_paul_MCMC/chain_pos_JWST_emission_petitRADTRANSpaper30.pickle'
with open(fname2, 'rb') as f:
    lines2 = pickle.load(f)
    l3= pickle.load(f)
    l4= pickle.load(f)
    l5= pickle.load(f)

sampls = l5

samples= sampls.reshape((240,50,3))

# parameters
f = plt.figure()
ax = f.add_subplot(111)
plt.title('MCMC 240 walkers, 50 iterations \n log_gamma')
res0= plt.plot(samples[:,:,0].T, '-', color='k', alpha=0.3)
plt.axhline(1.46347183e+00, color='red')
ax.get_yticklabels()[-1].set_color("red")
plt.text(52,1.46347183e+00,'1.46347183',rotation=0, c='red')
plt.show()

f = plt.figure()
ax = f.add_subplot(111)
plt.title('MCMC 240 walkers, 50 iterations \n T_internal')
res1= ax.plot(samples[:,:,1].T, '-', color='b', alpha=0.3)
ax.axhline(1.08669123e+03, color='black')
ax.get_yticklabels()[-1].set_color("red")
plt.text(52,1.08669123e+03,'1086.69123',rotation=0, c='red')
plt.show()

f = plt.figure()
ax = f.add_subplot(111)
plt.title('MCMC 240 walkers, 50 iterations \n T_equilibrium')
res2= plt.plot(samples[:,:,2].T, '-', color='r', alpha=0.3)
ax.axhline(1.98450655e+03, color='black')
plt.text(52,1.98450655e+03,'1984.50655',rotation=0, c='red')
plt.show()

tmp = corner.corner(sampls, labels=['log_gamma','T_int','T_equ'], 
                truths=[1.5, 750, 2000], show_titles=True)



