import torch
import torch.nn.functional as F
from torch.distributions import Normal
import utils

def log_prob_fn(model, train_loader, num_batches,device):

    log_prob = torch.tensor([0.0],requires_grad=True).to(device)
    log_like = torch.tensor([0.0]).to(device)
    
    for batch, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        log_prob_out, log_like_out = model.log_prob_func(x_train, y_train, num_batches)
        log_prob = log_prob + log_prob_out
        log_like = log_like + log_like_out

    if torch.cuda.is_available():
        del x_train, y_train
        torch.cuda.empty_cache()
    
    return log_prob, log_like

def sample_momentum(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = utils.flatten(model)
    mvn = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    #mvn = torch.distributions.MultivariateNormal(torch.zeros_like(params).to(device), torch.eye(len(params)).to(device))
    momentum = mvn.sample()
    return momentum

def hamiltonian(logprob, momentum):

    pe = - logprob
    ke = 0.5 * torch.dot(momentum, momentum)
    hamiltonian = ke + pe
    #print("pe", pe,"ke", ke,"ham", hamiltonian)

    return hamiltonian

def get_gradients(log_prob, model):
    """
    autograd.grad will compute and return the sum of gradients of outputs with respect to the inputs

    If RETURNs NONE => no gradient was computed which means change of params does not affect log_prob
    """

    # del H / del theta : differentiate log_prob w.r.t. params      
    grad = torch.autograd.grad(log_prob, model.parameters(), allow_unused=True)#, retain_graph=True)#, grad_outputs=None,,retain_graph=None,
                                 #create_graph=False)


    grad_flattened = torch.cat([grad[i].flatten() for i in range(6)])
    # del H / del m : differentiate KE w.r.t. momentum 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return grad_flattened #params

def get_predictions(model, test_loader, device):
    with torch.no_grad():
        accs = []
        for i, (x_test, y_test) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device)

            pred = model.eval(x_test, y_test)

            #print(y_test)
            acc = (pred.argmax(dim=-1) == y_test).to(torch.float32).mean()

            accs.append(acc.item())
            if(i==0):
                break
    return accs

class HMCsampler():

    def __init__(self, n_samples, step_size, N_leapfrog, burnin, device):

        
        self.n_samples = n_samples
        self.step_size = step_size
        self.N_leapfrog = N_leapfrog #int(traj_len // step_size + 1)
        self.burnin = burnin
        self.device = device



    def leapfrog(self, params, momentum, log_prob, N_leapfrog, model, train_loader, num_batches):
        #should return the whole trajectory of values or just the final values? - for now only final values
        # params = params.clone()
        # momentum = momentum.clone()
        
   

        #get initial grads
        grad_init = get_gradients(log_prob, model)
        print("initial grad", grad_init)

        #do half a momentum step before beginning loop
        momentum = momentum + 0.5 * self.step_size * grad_init
        
        for i in range(N_leapfrog):
            #update params by a full step
            params = params + self.step_size * momentum
            #calculate log_prob again, update model params before doing that
            utils.unflatten(params, model)
            #BUGg? log_prob_fn returns both log_prob and log_likelihood??
            log_prob, log_like = log_prob_fn(model, train_loader, num_batches, self.device)
            #get gradients after params have been updated
            grad_flattened = get_gradients(log_prob, model)
            #update momentum by a full step
            momentum = momentum + self.step_size * grad_flattened

        #update momentum by half a step at the end of the loop
        momentum = momentum + 0.5 * self.step_size * grad_flattened#.clone()

        prop_param, prop_momentum = params, momentum#.clone(), momentum#.clone()

        return prop_param, prop_momentum

    def metropolis(self, ham_old, ham_new):
        rho = min(0., -ham_new +  ham_old)

        if(rho>= torch.log(torch.rand(1)).to(self.device)):
            #print("Accept")
            return 0
            #see line 1000 in samplers.py
        else:
            #print("Reject")
            return 1

    def sample(self, model, train_loader, num_batches, device): 

        #params_init
        params = utils.flatten(model).clone().requires_grad_()

        N_leapfrog = self.N_leapfrog
        count = 0
        n_chain = self.n_samples - self.burnin
        num_params = len(params)
        samples = torch.zeros((n_chain, num_params))
        
        log_like_ret = torch.zeros((self.n_samples))#.requires_grad(True) #added later 
        log_prob_ret = torch.zeros((self.n_samples))

        for i in range(self.n_samples):    
            
            mom = sample_momentum(model)
            log_prob, log_like_ = log_prob_fn(model, train_loader, num_batches, device)
            ham_old = hamiltonian(log_prob, mom)

            params_new, mom_new = self.leapfrog(params, mom, log_prob, N_leapfrog, model, train_loader, num_batches)
            params_new = params_new.to(self.device).detach().requires_grad_()
            mom_new = mom_new.to(self.device)
            #log_prob needs to be recalculated everytime params are updated.
            #final log_prob/log_likelihood should be recorded after the proposed 
            #values have been accepted or rejected.
    
            utils.unflatten(params_new, model)

            log_prob_new, log_like_new = log_prob_fn(model, train_loader, num_batches, device)

            ham_new = hamiltonian(log_prob_new, mom_new)

            accept = self.metropolis(ham_old, ham_new)

            if (accept == 0):
                params = params_new.clone()
                mom = mom_new.clone()
                count += 1
                log_like_ret[i] = log_like_new
                log_prob_ret[i] = log_prob_new
                print("Accept")
            else:
                params = params.clone()
                mom = mom.clone()
                log_like_ret[i] = log_like_
                log_prob_ret[i] = log_prob

            utils.unflatten(params, model)

            #add sample values to the chain
            if(i>=self.burnin):
                samples[i-self.burnin][:] = params
            else:
                pass

        print("Acceptance ratio = ", count/self.n_samples)
        return samples, log_like_ret, log_prob_ret

#metrics

#check for convergence: R_hat
#compare within-chain variance averaged over multiple chains and between chain variance

# def potential_scale_reduction():



# def effective_sample_size():