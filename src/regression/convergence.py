



"""
Debugging convergence in Crocoddyl

"""

from valueNetwork import *
from residualNetwork import *
from dataset import Datagen
from plotTrajectories import plot_trajectories
from terminalUnicycle import ResidualUnicycle
import torch
import crocoddyl
import numpy as np
from unicycle_utils import *


# ddp.th_stop
STOP = 1e-8
# Maxiters 
MAXITERS = 1000


# Residual network trained on x ** 2
resnet = torch.load("rescostNet.pth")


# Random starting position generated from [2.1, 2.1] for x, y and [-2pi, 2pi] for theta
#xyz = np.array([np.random.uniform(-2.1, 2.1),
#                np.random.uniform(-2.1, 2.1),
#                np.random.uniform(-2*np.pi, 2*np.pi)])



xyz = np.array([-1.6012128 , -0.6732455 , -2.53071119])

model = crocoddyl.ActionModelUnicycle()
terminal_model = ResidualUnicycle(resnet)
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