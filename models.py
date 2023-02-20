import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Linear_HMC
dropout_rate1= 0.0
dropout_rate2= 0.0
dropout_rate3= 0.0

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        
        self.fc1 = nn.Linear (input_size*input_size, hidden_size)
        self.fc2 = nn.Linear (hidden_size, hidden_size)
        self.out = nn.Linear (hidden_size, output_size)
        self.imsize = input_size
        
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward (self, x):

        x = x.view(-1, self.imsize*self.imsize)
     
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)      
        
        x = self.out(x)

        #output = F.log_softmax(x, dim=1)
        return x
    
class MLPhmc(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, prior_var, prior_type):
        super().__init__()
        
        self.fc1 = Linear_HMC(input_size*input_size, hidden_size, prior_var, prior_type)
        self.fc2 = Linear_HMC(hidden_size, hidden_size, prior_var, prior_type)
        self.out = Linear_HMC(hidden_size, output_size, prior_var, prior_type)
        self.imsize = input_size
        self.out_dim = output_size
        # # weight initialisation:
        # # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

    def forward (self, x):

        x = x.view(-1, self.imsize*self.imsize)
     
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)      
        
        x = self.out(x)

        #output = F.log_softmax(x, dim=1)
        return x
    
    def log_prior(self):
        return self.fc1.log_prior + self.fc2.log_prior + self.out.log_prior
    
    def log_like(self, outputs, target):
        return F.cross_entropy(outputs, target, reduction='sum')
    
    # def log_post(self, log_prior, log_post):
    #     return self.fc1.log_pos
    
    def log_prob_func(self, input, target, samples = 1):

        #dont need multiple samples here right?
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        
        log_priors = torch.zeros(samples).to(self.device)
        #log_posts = torch.zeros(samples).to(self.device)
        log_likes = torch.zeros(samples).to(self.device)

        for i in range(samples):
            outputs[i] = self(input)

            log_priors[i] = self.log_prior()
            #log_posts[i] = self.log_post()

            log_likes[i] = self.log_like(outputs[i,:,:], target)

        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        #log_post = log_posts.mean()
        log_like = log_likes.mean()

        log_prob  = log_like + log_prior/2.0 #num_batches = 2 for now

        return log_prob
    
    def eval(self, input, target, samples = 1):

        #dont need multiple samples here right?
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        outputs = torch.zeros(samples, target.shape[0], self.out_dim).to(self.device)
        
        log_priors = torch.zeros(samples).to(self.device)
        #log_posts = torch.zeros(samples).to(self.device)
        log_likes = torch.zeros(samples).to(self.device)

        for i in range(samples):
            outputs[i] = self(input)

            log_priors[i] = self.log_prior()
            #log_posts[i] = self.log_post()

            log_likes[i] = self.log_like(outputs[i,:,:], target)

        # the mean of a sum is the sum of the means:
        log_prior = log_priors.mean()
        #log_post = log_posts.mean()
        log_like = log_likes.mean()

        log_prob  = log_like + log_prior/2.0 #num_batches = 2 for now

        return outputs
    



class MLPdropout(nn.Module):
    
    def __init__(self, input_size, hidden_size,output_size):
        super().__init__()
        
        self.dropout1 = nn.Dropout(p=dropout_rate1)
        self.fc1 = nn.Linear (input_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout_rate2)
        self.fc2 = nn.Linear (hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout_rate3)
        self.out = nn.Linear (hidden_size, output_size)
        
        # weight initialisation:
        # following: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward (self, x):
        
        x = self.dropout1(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)       
        
        x = self.out(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class LeNet(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size=5):
        
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size, padding =1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size, padding =1)
        #use conv 3 and conv4 with mirabest
        #self.conv3 = nn.Conv2d(16, 26, kernel_size, padding =1)
        #self.conv4 = nn.Conv2d(26, 32, kernel_size, padding =1)
        
        self.fc1   = nn.Linear(5*5*16, 120) #--> MNIST LeNet without padding
        #self.fc1   = nn.Linear(36*36*16, 120) #--> MiraBest LeNet with padding
        #self.fc1   = nn.Linear(7*7*32, 120) #mirabest lenet + 2 more conv layers
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, output_channels)
        self.dropout1 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        
        '''
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = F.max_pool2d(x, 2)
        '''
        
        #print(x.shape)
        #x = x.view(x.size()[0], -1)
        x = x.view(-1, 5*5*16) #--> MNIST with padding
        #x = x.view(-1, 4*4*16) #--> MNIST without padding
        #x = x.view(-1, 7*7*32) #-->MiraBest LeNet with padding

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return(x)
    
    
