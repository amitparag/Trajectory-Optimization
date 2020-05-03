import torch
import crocoddyl
import numpy as np
from residualNetwork import ResidualNet
from valueNetwork import ValueNet
def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()

class TerminalUnicycle(crocoddyl.ActionModelAbstract):
    
    def __init__(self, net):
       
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = net
        
    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        
        x = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        with torch.no_grad():
            data.cost = self.net(x).item()

    def calcDiff(self, data, x, u=None):        
        if u is None:
            u = self.unone
        
        x0 = torch.as_tensor(m2a(x), dtype = torch.float32).resize_(1, 3)
        j = self.net.jacobian(x0).detach().numpy()
        h = self.net.hessian(x0).detach().numpy()
        data.Lx = a2m(j)
        data.Lxx = a2m(h) 