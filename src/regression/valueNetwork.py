
import torch 
import numpy as np
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 1,
                 nhiddenunits:int = 64):
        
        super(ValueNet, self).__init__()
        
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
        print(self, self.device)
        

    def forward(self, x):
        
        x1 = self.activation(self.fc1(x)) 
        x2 = self.activation(self.fc2(x1)) 
        x3 = self.fc3(x2) 
        
        return x3
    

    def jacobian(self, x):
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j

    def hessian(self, x):
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h

    def batch_hessian(self, x):
        h = [torch.autograd.functional.hessian(self.forward, x) for x in x]
        return torch.stack(h).squeeze()
    
    def batch_jacobian(self, x):
        j = [torch.autograd.functional.jacobian(self.forward, x) for x in x]
        return torch.stack(j).squeeze()
            
