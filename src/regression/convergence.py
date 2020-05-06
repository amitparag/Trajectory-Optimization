


"""
Debugging convergence in Crocoddyl

"""

from valueNetwork import *
from residualNetwork import *
from dataset import Datagen
from plotTrajectories import plot_trajectories
from terminalUnicycle import TerminalUnicycle
import torch
import crocoddyl
import numpy as np
from unicycle_utils import *


# ddp.th_stop
STOP = 1e-8
# Maxiters 
MAXITERS = 10000



#fnet = torch.load("valueNet.pth") # Feed Forward net trained on ddp.cost
#lnet = torch.load("costNet.pth")  # Feed Forward net trained on x**2
rnet = torch.load("resNet.pth")   # Residual net trained on  ddp.cost  


# Random starting position generated from [2.1, 2.1] for x, y and [-2pi, 2pi] for theta
#xyz = np.array([np.random.uniform(-2.1, 2.1),
#                np.random.uniform(-2.1, 2.1),
#                np.random.uniform(-2*np.pi, 2*np.pi)])

xyz = np.array([ 1.06050777,  2.06662238, -5.95716119])

model = crocoddyl.ActionModelUnicycle()
terminal_model = TerminalUnicycle(rnet)
model.costWeights = np.matrix([1,1]).T
problem = crocoddyl.ShootingProblem((xyz).T, [ model ] * 30, terminal_model)
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
ddp.th_stop = STOP
ddp.solve([] , [], MAXITERS)

mt = problem.terminalModel
dt = problem.terminalData
xt = ddp.xs[-1]


print(dt.Lxx)
print(ddp.iter)