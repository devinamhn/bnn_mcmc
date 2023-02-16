import torch
import torch.nn.functional as F
from torch.distributions import Normal

#this can be defined in the model file as before
def log_like(outputs,target):
    #log P(D|w)
    return F.cross_entropy(outputs, target, reduction='sum')

def log_prior(params, var):
    mean = 
    stddev = 
    torch.distributions.Normal(mean, stddev).log_prob(params).sum()

    lnP = torch.distributions.Normal(0,var).log_prob(x)

# def log_prob_func(lo ):

#     log_likelihood = self.log_like(outputs[i,:,:], target)

#     #priors from VI code
#     #log_prior = Gaussian()

#     #log_prob = log_likelihood + log_prior

#     return


############################################################################
# See where make_lenet5_fun is used
# last layer is just nn.Linear(num_classes), no softmax so they must be using crossentropy loss
# returning logits in resnet

#user defined params

step_size = 1e-1 
#trajectory length = length of HMC simulation for each iteration
alpha_prior = 1 #for now to match the code
trajectory_len = torch.pi * alpha_prior/ 2 #alpha_prior = std of prior distribution

N_leapfrog = int(trajectory_len // step_size + 1)
n_samples = 100

class HMCsampler():

    def __init__(self, params, n_samples, step_size, traj_len, ):

        self.params = params 
        self.n_samples = n_samples
        self.N_leapfrog = int(traj_len // step_size + 1)



    def sample_momentum(params): #input flat vector of model params
        '''
        for each param value, a momentum variable is defined
        momentum is sampled from a multivariate normal distribution 
        '''
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(params), torch.eye(len(params)))
        return mvn.sample()

    def get_gradients(self, log_prob, params):
 
        #call params.backwards() to populate params.grad()

        # del H / del theta : differentiate log_prob w.r.t. params  
        params.grad = torch.autograd.grad(log_prob, params)#[0]

        # del H / del m : differentiate KE w.r.t. momentum 


        return params

    #update position and momentum variables
    def updates(self, init_values):

        params, momentum = init_values
        grad = self.get_gradients(params).grad
        #lambda functions : momentum and grad are values to the fn

        #update momentum by half a step size
        momentum = momentum + 0.5 * step_size * grad
        #momentum = jax.tree_map(lambda m, g: m + 0.5 * step_size * g, momentum, grad)
        

        #update position aka param values by a full step size using the updated momentum
        
        params = params + step_size * momentum
        #params = jax.tree_map(lambda s, m: s + step_size*m, params, momentum)

        #calculate posterior log_prob, gradient of posterior log_prob, loglikelihood, net_state

        #update momentum again by half a step 
        
        momentum = momentum + 0.5*grad
        #momentum = jax.tree_map(lambda m, g: m + 0.5* step_size*g, momentum, grad)

        #return updated params and momentum values
        #return params, net_state, momentum, grad, log_prob, log_likelihood

    #repeat updates step for number of leapfrog steps

    def leapfrog(self, init_values, N_leapfrog):
        #should return the whole trajectory of values or just the final values?

        for i in range(N_leapfrog):
            
            proposed_values = self.updates(init_values)

        return proposed_values


    #log_prob_func - where is this defined?? - was mixing up two code repos
    def calculate_hamiltonian(self, params, momentum, log_prob_func ):
        #calculate potential and kinetic energies

        log_prob = log_prob_func(params)

        potential_energy = -log_prob #is a normalizing constant reqd

        #assuming mass = 1
        kinetic_energy = 0.5 * torch.dot(momentum, momentum)

        hamiltonian = kinetic_energy + potential_energy

        return hamiltonian




    def metropolis_correction(self, H_old, H_new):

        #calculate log acceptance ratio using prev and new values of hamiltonian

        #ratio = H_new/H_old 

        accept_prob = torch.min(1., torch.exp(H_new - H_old))

        #sample from a random distribution
        torch.rand(1)

        
        return 

    def sample(self, num_samples, params):

        for i in range(num_samples):

            momentum = self.sample_momentum(params)
            hamiltonian = self.calculate_hamiltonian(params, momentum, log_prob_func)

            traj_params, traj_momentum = self.leapfrog(init_values, N_leapfrog)

            #update with new values
            params = traj_params[-1].requires_grad_()
            momentum = traj_momentum[-1]

            new_hamiltonian = self.calculate_hamiltonian(params, momentum, log_prob_func)

            ratio = self.metropolis_correction(hamiltonian, new_hamiltonian)

         



#metrics

#check for convergence: R_hat
#compare within-chain variance averaged over multiple chains and between chain variance

# def potential_scale_reduction():



# def effective_sample_size():