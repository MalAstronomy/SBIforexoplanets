import os
import sys
sys.path.insert(1, '/home/mvasist/constraining-dark-matter-with-stellar-streams-and-ml/notebooks/')

import argparse
import hypothesis
hypothesis.disable_gpu()  # You can change this
import matplotlib.pyplot as plt
import numpy as np
import torch
# from sbi import utils as utils

from hypothesis.stat import highest_density_level
from util import MarginalizedAgePrior
from util import Prior
from scipy.stats import chi2
from util import load_ratio_estimator

from tqdm import tqdm
from util import download
from util import load

@torch.no_grad()
def Prior():
    lower = torch.tensor([0., -4 , 2 ]).float()
    lower = lower.to(hypothesis.accelerator)
    upper = torch.tensor([2000., 0, 3.7 ]).float()
    upper = upper.to(hypothesis.accelerator)

    return torch.distributions.uniform.Uniform(lower, upper)  #BoxUniform_new(normal, uniform)

class Coverage_class:
    
    def __init__(self, name=''):
        self.name = name
        self.prior = Prior()    
    
    @torch.no_grad()
    def highest_density_level(self, density, alpha, min_epsilon=10e-16, region=False):
        # Check if a numpy type has been specified
        if type(density).__module__ != np.__name__:
            density = density.cpu().clone().numpy()
        else:
            density = np.array(density)
        density = density.astype(np.float64)
        # Check the discrete sum of the density (for scaling)
        integrand = density.sum()
        density /= integrand
        # Compute the level such that 1 - alpha has been satisfied.
        optimal_level = density.max()
        epsilon = 10e-00  # Current error
        while epsilon >= min_epsilon:
            optimal_level += 2 * epsilon  # Overshoot solution, move back
            epsilon /= 10
            area = 0.0
            while area < (1 - alpha):
                area_under = (density >= optimal_level)
                area = np.sum(area_under * density)
                optimal_level -= epsilon  # Gradient descent to reduce error
        # Rescale to original
        optimal_level *= integrand
        # Check if the computed mask needs to be returned
        if region:
            return optimal_level, area_under
        else:
            return optimal_level
        
    @torch.no_grad()
    def compute_log_posterior(self, r, observable, resolution= 10 , extent=[0, 2000, -4, 0, 2, 3.7]):
        prior = Prior()
        # Prepare grid
        epsilon = 0.00001
        resolution = 10
        p1 = torch.linspace(extent[0], extent[1]-epsilon, resolution)  # Account for half-open interval of uniform prior
        p2 = torch.linspace(extent[2], extent[3]-epsilon, resolution)  # Account for half-open interval of uniform prior
        p3 = torch.linspace(extent[4], extent[5]-epsilon, resolution)  # Account for half-open interval of uniform prior ###
        p1 = p1.to(hypothesis.accelerator)
        p2 = p2.to(hypothesis.accelerator)
        p3 = p3.to(hypothesis.accelerator) #####
#         print(p1)
#         print(p1.size())
        g1, g2, g3 = torch.meshgrid(p1.view(-1), p2.view(-1), p3.view(-1)) ######
#         print(g1.size())
        # Vectorize
        inputs = torch.cat([g1.reshape(-1, 1), g2.reshape(-1, 1), g3.reshape(-1, 1)], dim=1) #####
        log_prior_probabilities = self.prior.log_prob(inputs).sum(axis=1).view(-1, 1)
#         print(log_prior_probabilities.size())#, log_prior_probabilities)
#         print('ob ', observable.size())
        observables = observable.repeat(resolution ** 3, 1).float() #######
        observables = observables.to(hypothesis.accelerator)
#         print(observables.size(), inputs.size())
        #observables = observables.view(-1, 371)   # for mlp only
#         print(observables.size(), inputs.size())
        log_ratios = r._classifier_logits(inputs, observables, num_atoms=2)
        #log_posterior = (log_prior_probabilities + log_ratios).view(resolution, resolution, resolution).cpu()
        #print(len(inputs))
        log_posterior = (log_prior_probabilities + log_ratios[:len(inputs)]).view(resolution, resolution, resolution).cpu() ##   
        #print(log_posterior.size())
        return log_posterior
    
    @torch.no_grad()
    def compute_log_pdf(self, r, inputs, outputs):
        inputs = inputs.to(hypothesis.accelerator)
        outputs = outputs.to(hypothesis.accelerator)
        
        log_ratios = r._classifier_logits(inputs.repeat(2,1), outputs.repeat(2,1), num_atoms=2) #################
        log_ratios = log_ratios[:len(inputs)]
        #print('lr: ', log_ratios)
        log_prior = self.prior.log_prob(inputs).sum(axis=1)    
        #print('lpr: ', log_prior)
        return (log_prior + log_ratios).squeeze()
    
    @torch.no_grad()
    def coverage(self, r, inputs, outputs, confidence_level=0.95, resolution=10, extent= [0, 2000, -4, 0, 2, 3.7]):
        n = len(inputs)
        covered = 0
        alpha = 1.0 - confidence_level
        for index in tqdm(range(n), "Coverages evaluated"):
            # Prepare setup
            nominal = inputs[index].squeeze().unsqueeze(0)
            observable = outputs[index].squeeze().unsqueeze(0)
        
            nominal = nominal.to(hypothesis.accelerator)
            observable = observable.to(hypothesis.accelerator)
            pdf = self.compute_log_posterior(r, observable, resolution=resolution, extent=extent).exp().view(resolution, resolution, resolution) ############
#             print('pdf ', pdf)
#             print('im outside')
            nominal_pdf = self.compute_log_pdf(r, nominal, observable).exp()
            level = self.highest_density_level(pdf, alpha)
            if nominal_pdf >= level:
                covered += 1    
                
        return covered / n

        
###############################################################################################################################
###############################################################################################################################

# For one parameter
    
#     @torch.no_grad()
#     def compute_log_posterior(self, ratio_estimator, observable, resolution=100):
#         observable = observable.view(1, -1)
#         # Since we marginalize over the age, we only need a grid over the WDM mass.
#         inputs = torch.linspace(1.0, 50.0, resolution).view(-1, 1)  # In KeV
#         # Because I'm vectorizing the computation, we need to repeat the observable.
#         outputs = observable.repeat(resolution, 1)
#         log_ratios = ratio_estimator.log_ratio(inputs=inputs, outputs=outputs)
#         #print('clp ip, op: ', inputs.size(), outputs.size(), log_ratios.size())
#         log_prior = 0.0  # I'm a bit lazy here, but here you should include something like `prior.log_prob(inputs)`
#         log_posterior = log_ratios + log_prior  # Because: posterior = prior * likelihood-to-evidence ratio
#                                                 #    -> log posterior = log prior + log likelihood-to-evidence ratio
#         return log_posterior

#     @torch.no_grad()
#     def compute_log_pdf(self, r, inputs, outputs):
#         inputs = inputs.view(-1, 1)  # Ensure correct shape
#         log_ratios = r.log_ratio(inputs=inputs, outputs=outputs)
#         #print('clpdf ip, op: ', inputs.size(), outputs.size(), log_ratios.size())
#         # log_prior = prior.log_prob(inputs)
#         log_prior = 0.0

#         return (log_prior + log_ratios).squeeze()

#     @torch.no_grad()
#     def covered(self, r, generating_parameter, stream, confidence_level=0.95):
#         pdf = self.compute_log_posterior(r, stream).exp()
#         posterior_density_of_generating_parameter = self.compute_log_pdf(r, generating_parameter, stream).exp()
#         alpha = 1.0 - confidence_level
#         level = self.highest_density_level(pdf, alpha)
#         if posterior_density_of_generating_parameter >= level:  # If the density of the generating parameters is larger compared to the level of the hyper-plane, then the interval covers the generating parameter.
#             covered = True
#         else:
#             covered = False

#         return covered

#     @torch.no_grad()
#     def coverage(self, r, inputs, outputs, confidence_level):
#         n = len(inputs)
#         num_covered = 0
#         for index in tqdm(range(n), "Coverages evaluated"):
#             # Prepare setup
#             nominal = inputs[index].squeeze().unsqueeze(0)
#             observable = outputs[index].squeeze().unsqueeze(0)
#             if self.covered(r, nominal, observable, confidence_level):
#                 num_covered += 1

#         return num_covered / n
    
###############################################################################################################################
        
    