import torch
import torch.nn.functional as F
from torch.distributions import Normal
from utils import *

def log_prob_fn(model, train_loader, device):
    log_prob = torch.tensor([0])
    for batch, (x_train, y_train) in enumerate(train_loader):
        
        x_train, y_train = x_train.to(device), y_train.to(device)
        #outputs = model.forward(x_train)
        log_prob = log_prob + model.log_prob_func(x_train, y_train)
        #print(batch)

        if(batch == 1):
            break
    return log_prob

def sample_momentum(model):
    params = flatten(model)
    mvn = torch.distributions.MultivariateNormal(torch.zeros_like(params), torch.eye(len(params)))
    momentum = mvn.sample()
    return momentum

def hamiltonian(logprob, momentum):

    pe = - logprob
    ke = 0.5 * torch.dot(momentum, momentum)
    hamiltonian = ke + pe
    #print("pe", pe,"ke", ke,"ham", hamiltonian)

    return hamiltonian

def get_gradients(log_prob, model):
 
    #call params.backwards() to populate params.grad()

    # del H / del theta : differentiate log_prob w.r.t. params  
    grad = torch.autograd.grad(log_prob, model.parameters(), allow_unused=True, retain_graph=True)#, grad_outputs=None,,retain_graph=None,
                                 #create_graph=False)
    # print("GRADIENT")
    # print(torch.cat([grad[i].flatten() for i in range(6)]))
    grad_flattened = torch.cat([grad[i].flatten() for i in range(6)])

    # del H / del m : differentiate KE w.r.t. momentum 


    return grad_flattened #params
class HMCsampler():

    def __init__(self, n_samples, step_size, N_leapfrog):

        
        self.n_samples = n_samples
        self.step_size = step_size
        self.N_leapfrog = N_leapfrog #int(traj_len // step_size + 1)
        #self.device = device


    # def sample_momentum(params): #input flat vector of model params


    # #log_prob_func - where is this defined?? - was mixing up two code repos
    # def calculate_hamiltonian(self, params, momentum, log_prob_func):

    # def get_gradients(self, log_prob, params):
 

    #update position and momentum variables
    
    #should momentum be sampled inside the udpate function?
    def updates(self, params, momentum, log_prob, model): #DOES NOT REQUIRE params

        #step_size = 1e-1
        grad_flattened = get_gradients(log_prob, model)
        params = flatten(model)
        # print("MOMEMTUM BEFORE UPDATE")
        # print(momentum)
        momentum = momentum + 0.5 * self.step_size * grad_flattened
    
        # print("PARAMS BEFORE UPDATE")
        # print(params)
        params = params + self.step_size * momentum
        # print("PARAMS")
        # print(params)
        # print("MOMENTUM")
        momentum = momentum + 0.5*grad_flattened
        # print(momentum)
        # print("END UPDATE")

        #not a pure function, modifying the args
        return params, momentum 


    def leapfrog(self, params, mom, log_prob, N_leapfrog, model):
        #should return the whole trajectory of values or just the final values?
        #the whole trajectory of values or just the final values?

        for i in range(N_leapfrog):
            proposed_values = self.updates(params, mom, log_prob, model)
            mom =  sample_momentum(model)    

        return proposed_values

    def metropolis(self, ham_old, ham_new):

        #print(ham_old, ham_new, ham_new - ham_old)

        #print(ham_new/ham_old)

        rho = min(0., -ham_new +  ham_old)
        print(rho)
        
        if(rho>= torch.log(torch.rand(1))):
            print("Accept")
            return 0

            #see line 1000 in samplers.py
        else:
            print("Reject")
            return 1

    def sample(self, model, train_loader, device):
        
        log_prob = log_prob_fn(model, train_loader, device)

        mom = sample_momentum(model)
        params = flatten(model)
        ham_old = hamiltonian(log_prob, mom)

        N_leapfrog=20
        count = 0
        for i in range(self.n_samples):
            #for now - 
            #leapfrog only returns the final value not the whole trajectory of params and momentum
            
            #mom =  momentum(model)
            params_new, mom_new = self.leapfrog(params, mom, log_prob, N_leapfrog, model)

            log_prob = log_prob_fn(model, train_loader, device)
            ham_new = hamiltonian(log_prob, mom_new)

            accept = self.metropolis(ham_old, ham_new)
            if (accept == 0):
                params = params_new
                mom = mom_new
                count += 1
            else:
                params = params
                mom = mom
                
            print(params)
            with torch.no_grad(): #is it correct to us this here?
                params.copy_(params)

        print("Acceptance ratio = ", count/self.n_samples)
        return params

#metrics

#check for convergence: R_hat
#compare within-chain variance averaged over multiple chains and between chain variance

# def potential_scale_reduction():



# def effective_sample_size():