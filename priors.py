import torch
#import numpy as np

class GaussianPrior():

    def __init__(self,var):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.var = torch.Tensor(var).to(self.device)

    def log_prob(self,x):

        lnP = torch.distributions.Normal(0,self.var).log_prob(x)
 
        return lnP

# -----------------------------------------------------------------------------
#log P(w) = prior on the weights/biases
class GMMPrior():

    def __init__(self,parms):

        if len(parms)!=3:
            print("Incorrect dimensions for prior parameters")
        else:
            self.pi, self.stddev1, self.stddev2 = parms

    def log_prob(self,x):

        var1 = torch.pow(self.stddev1,2)
        var2 = torch.pow(self.stddev2,2)
        prob = (torch.exp(-torch.pow((x-0),2)/(2*var1))/(self.stddev1*torch.sqrt(2*torch.pi)))*self.pi
        prob+= (torch.exp(-torch.pow((x-0),2)/(2*var2))/(self.stddev2*torch.sqrt(2*torch.pi)))*(1-self.pi)
        
        logprob = torch.log(prob)

        return logprob

#------------------------------------------------------------------------------

class LaplacePrior():
    
    def __init__(self,parms):

        self.mu, self.b = parms
    
    def log_prob(self, x):
        
        logprob = torch.log(1/(2*self.b)) - abs(x - self.mu)/self.b
    
        return logprob
    
class LaplaceMixture():
    
    def __init__(self,parms):

        self.pi, self.b1, self.b2 = parms
    
    def log_prob(self, x):
        mu = 0
        prob = self.pi * (1/(2*self.b1)* torch.exp( -abs(x - mu)/self.b1))
        prob+= (1-self.pi) * (1/(2*self.b2)* torch.exp( - abs(x - mu)/self.b2))
        logprob = torch.log(prob)
    
        return logprob
#------------------------------------------------------------------------------
#horseshoe prior
class CauchyPrior():
    pass
#------------------------------------------------------------------------------
#noise contrastive prior