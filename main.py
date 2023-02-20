import torch
from datamodules import MNISTDataModule, MiraBestDataModule
from models import MLP, MLPhmc
from utils import *
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
#data imports
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from hmc import HMCsampler

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#check if a GPU is available:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)


# config_dict, config = parse_config('config_hmc.txt')
# print(config_dict, config)

datamodule = MNISTDataModule(128)
batch = 1
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
 
# train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
# test_loader = datamodule.test_loader()

#model = MLP(28, 2, 10).to(device)
model = MLPhmc(28, 2, 10, [1], 'Gaussian').to(device)
print(summary(model, input_size=(1, 28, 28)))

n_samples = 10
step_size = 1e-1
N_leapfrog = 20

sampler = HMCsampler(n_samples,step_size, N_leapfrog)

params = sampler.sample(model, train_loader, device)

with torch.no_grad():
    accs = []
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        pred = model.eval(x_test, y_test)

        print(y_test)
        acc = (pred.mean(dim=0).argmax(dim=-1) == y_test).to(torch.float32).mean()

        accs.append(acc.item())
        if(i==0):
            break

print("accuracy", np.mean(accs)*100, "%")
# log_prob = log_prob_fn(model, train_loader, device)

# mom = momentum(model)
# params = flatten(model)
# ham_old = hamiltonian(log_prob, mom)
# N_leapfrog=20

# params, mom = leapfrog(params, mom, log_prob, N_leapfrog, model)
# print(params, mom)

# log_prob = log_prob_fn(model, train_loader, device)

# #dont need these 2 lines - params, mom will be the new values
# #mom = momentum(model) 
# #params = flatten(model)

# ham_new = hamiltonian(log_prob, mom)

# print(ham_old, ham_new, ham_new - ham_old)

# print(ham_new/ham_old)

# rho = min(0., -ham_new +  ham_old)
# print(rho)

# if(rho>= torch.log(torch.rand(1))):
#     print("Accept")

#     #see line 1000 in samplers.py
# else:
#     print("Reject")

exit()



# mass = 1.0
# params_init = params
# inv_mass = torch.ones(params_init.shape) / mass