
"""
Solver class to solve crocoddyl to return
    1: ddp
    2: value
    3: diagonal of hessian
    4: diagonal of gradient

"""


import numpy as np
import crocoddyl

class Solver:
    def ddp(self, xyz):
        """
        Returns the ddp after solving crocoddyl with the given starting position
        """
        model = crocoddyl.ActionModelUnicycle()
        T = 30
        model.costWeights = np.matrix([1,1]).T
        problem = crocoddyl.ShootingProblem(np.array(xyz).T, [ model ] * T, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        return ddp
    
    def value_function(self, test):
        """
        Returns the value associated with xyz after solving crocoddyl
        """
        if not isinstance(test, np.ndarray):
            test = np.array(test).reshape(1, -1)
            ddp = self.ddp(test)
            return np.array([ddp.cost])

        else:
            values = []
            for xyz in test:
                ddp = self.ddp(xyz)
                values.append([ddp.cost])
            return np.array(values)
        
        
    def gradients(self, test):
        """
        @returns the diagonals of the gradient, hessian
        """
        print("Getting Vx[1], Vxx[11] from crocoddyl")
        diff1 = []
        diff2 = []
        for xyz in test:
            ddp = self.ddp(xyz)
            diff1.append(ddp.Vx[0][1])
            diff2.append(ddp.Vxx[0][1][1])
            
        return np.array(diff1), np.array(diff2)
        