import numpy as np
import torch
import torch.nn as nn

class ResidualNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 3,
                 nhiddenunits:int = 128):
        
        super(ResidualNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.nhiddenunits = nhiddenunits
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.nhiddenunits)
        self.fc2 = nn.Linear(self.nhiddenunits, self.nhiddenunits)
        self.fc3 = nn.Linear(self.nhiddenunits, self.out_features)

        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
        # Activation
        self.activation = nn.Tanh()
      
        self.device = torch.device('cpu')
        self.to(self.device)
        print(self)
        

    def forward(self, x):
        
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x)) 
        x = self.fc3(x) 
        return x.pow(2).sum(dim=1, keepdim=True)


    def jacobian(self, x):
        x = x.reshape(1, 3)
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j
    
    def hessian(self, x):
        x = x.reshape(1, 3)
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h

    def batch_hessian(self, x):
        h = []
        for xyz in x:
            h.append(self.hessian(xyz))
        return torch.stack(h).squeeze()
    
    def batch_jacobian(self, x):
        j = []
        for xyz in x:
            j.append(self.jacobian(xyz))
        return torch.stack(j).squeeze()
    
    #......Gauss Approximation
    def residual(self, x):
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x)) 
        x = self.fc3(x)
        return x
    
    def jacobian_residual(self, x):
        j = torch.autograd.functional.jacobian(self.residual, x).squeeze()
        return 2 * j

    def gradient(self, x):
        j = self.jacobian_residual(x)
        r = self.residual(x)
        return j.T @ r


    def gauss_hessian(self, x):
        j = self.jacobian_residual(x).detach()
        return j.T @ j
    
    def batch_gradient(self, x):
        jj = []
        for x in x:
            
            j = self.gradient(x)
            jj.append(j)
        return torch.stack(jj).squeeze()
    
    def batch_gauss_hessian(self, x):
        jj = []
        for x in x:
            j = self.gauss_hessian(x).squeeze()
            jj.append(j)
        return torch.stack(jj).squeeze()
    
    