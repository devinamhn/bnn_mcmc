import torch
from datamodules import MNISTDataModule, MiraBestDataModule
from models import MLP
from utils import *
from torchsummary import summary
import matplotlib.pyplot as plt

#data imports
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from pytorch_lightning.demos.mnist_datamodule import MNIST
from torch.utils.data import DataLoader, random_split

#from hmc import log_like,log_prob_func, HMCsampler

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# config_dict, config = parse_config('config_hmc.txt')
# print(config_dict, config)

# datamodule = MiraBestDataModule(config_dict, config)
# train_loader, validation_loader, train_sampler, valid_sampler = datamodule.train_val_loader()
# test_loader = datamodule.test_loader()

model = MLP(28, 64, 10)
print(summary(model, input_size=(1, 28, 28)))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST('./' , train = True, download = True, transform = transform)
mnist_test = MNIST('./', train = False, download = True, transform = transform)

# plt.imshow(mnist_train.data[10].reshape((28,28)))
# plt.show()

N_tr = 100
N_val = 1000

x_train = mnist_train.data[:N_tr]
x_train = x_train[:,None]
y_train = mnist_train.targets[:N_tr]#.reshape((-1,1)).float()

x_val = mnist_train.train_data[N_tr:N_tr+N_val]#.float()/255.
x_val = x_val[:,None]
y_val = mnist_train.targets[N_tr:N_tr+N_val]#.reshape((-1,1)).float()


