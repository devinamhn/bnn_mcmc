###layers###
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
import numpy as np
from priors import GaussianPrior, GMMPrior, LaplacePrior, CauchyPrior, LaplaceMixture

class Linear_HMC(nn.Module):

    """
        Layer of our BNN.
    """

    def __init__(self, input_features, output_features, prior_var, prior_type):
        super().__init__()

        #set dim
        self.input_features = input_features
        self.output_features = output_features
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialize weight params
        self.w = nn.Parameter(torch.zeros(output_features, input_features).uniform_(-0.1, 0.1).to(self.device))
        
        #initialize bias params
        self.b =  nn.Parameter(torch.zeros(output_features).uniform_(-0.1, 0.1).to(self.device))

        # initialize prior distribution
        if(prior_type == 'Gaussian'):
            self.prior = GaussianPrior(prior_var) #1e-1
        elif(prior_type == 'GaussianMixture'):
            self.prior = GMMPrior(prior_var)
        elif(prior_type == 'Laplacian'):
            self.prior = LaplacePrior(prior_var)
        elif(prior_type == 'LaplaceMixture'):
            self.prior = LaplaceMixture(prior_var)
        elif(prior_type == 'Cauchy'):
            pass
            self.prior = CauchyPrior(prior_var)
        else:
            print("Unspecified prior")
       
        
    def forward(self, input):
        
        #record prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior) 
        
       
        #record posterior 
        #update with samples from the mcmc
        
        #why is this even needed then????? probably cna do without it because no distributions are being sampled
        self.w_post = self.w
        self.b_post = self.b

        #could use log_prob cause it was calculating it for a 
        #sample from the Normal dist.

        #self.log_post = self.log_prob(self.w).sum() + self.log_prob(self.b).sum()
        

        
        return F.linear(input, self.w, self.b)