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
from hmc import HMCsampler, get_predictions

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
num_batches= 1
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
 
# train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
# test_loader = datamodule.test_loader()

#model = MLP(28, 2, 10).to(device)
model = MLPhmc(28, 20, 10, [1], 'Gaussian').to(device)
print(summary(model, input_size=(1, 28, 28)))

n_samples = 300
step_size = 2e-4 #1e-3 #1e-1
N_leapfrog = 50 #20
burnin = 100

sampler = HMCsampler(n_samples,step_size, N_leapfrog, burnin, device)

samples, log_like, log_prob = sampler.sample(model, train_loader, num_batches, device)

accs = get_predictions(model, test_loader, device)
#print(samples)
print("accuracy", np.mean(accs)*100, "%")

params_len = len(samples[0][:])
#print(samples.reshape((1606, n_samples-burnin)))
samples = samples.reshape((params_len, n_samples-burnin))
torch.save(samples,'./results/test/samples.pt')
torch.save(log_like,'./results/test/log_like.pt')

plt.figure(dpi=200)
plt.plot(log_like.detach().numpy())
plt.savefig("./results/test/log_like.png")


torch.save(log_prob,'./results/test/log_prob.pt')

plt.figure(dpi=200)
plt.plot(log_prob.detach().numpy())
plt.savefig("./results/test/log_prob.png")


plt.figure(dpi=200)
plt.plot(samples[0][:].detach().numpy())
plt.savefig("./results/test/asample.png")

#exit()

# mass = 1.0
# params_init = params
# inv_mass = torch.ones(params_init.shape) / mass