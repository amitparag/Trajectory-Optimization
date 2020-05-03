"""

Datagen class to generate various types of data required to run experiments
See methods for more details.

"""


import numpy as np
import torch 
import torch.nn as nn
import crocoddyl
from solver import Solver
class Datagen:

    def positions_values(self, size:int = 4000, as_tensor:bool=True):
        
        """
        Solves crocoddyl to get values
        """
        print("Solving crocoddyl")
        solve = Solver()
        positions = self.random_xyz(size = size)
        values = solve.value_function(positions)

        if as_tensor:
            positions = torch.tensor(positions, dtype = torch.float32)
            values    = torch.tensor(values, dtype = torch.float32)

        return positions, values

    def positions_costs(self, size:int = 4000, as_tensor:bool=True):
        print("Solving sum(x ** 2")
        positions = self.random_xyz(size = size)
        costs     = self.cost_function(positions)
        if as_tensor:
            positions = torch.tensor(positions, dtype = torch.float32)
            costs     = torch.tensor(costs, dtype = torch.float32)

        return positions, costs

    
    def cost_function(self, xyz):
        """
        Returns the cost of a given array or list
        cost = sum( xyz ** 2)
        """
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz).reshape(1, -1)
        return np.sum(np.power(xyz, 2), axis  = 1, keepdims=True)



    def random_xyz(self,
                   size:int = 3000,
                   xlim = [-2.1,2.1],
                   ylim = [-2.1,2.1],
                   zlim = [-2*np.pi,2*np.pi]
                   ):

        """
        @returns: array of random data generated from the given ranges
        
        """

        min_x, max_x = xlim
        min_y, max_y = ylim
        min_z, max_z = zlim

        print("Generating random xyz from: \n ")
        print(f"  x = [ {min_x} , {max_x} ] \n")
        print(f"  y = [ {min_y} , {max_y} ]\n")
        print(f"  z = [ {min_z} , {max_z} ]\n")

        x = np.random.uniform(min_x, max_x, size = (size, 1))
        y = np.random.uniform(min_y, max_y, size = (size, 1))
        z = np.random.uniform(min_z, max_z, size = (size, 1))
        
        return np.hstack((x, y, z))   
    
    def grid_data(self,
                  size:int = 30,
                  ranges = [-1., 1.]
                  ):

        """
        @params:
        size = number of grid points
        @returns:
        grid array        
        """
        min_x, max_x = ranges
        xrange = np.linspace(min_x,max_x,size)
        data = np.array([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ])
        return data


    def circular_data(self, r=[2], n=[100]):
        """
        @params:
            r = list of radii
            n = list of points required from each radius
        @returns:
            array of points from the circumference of circle of radius r centered on origin
        Usage: circle_points([2, 1, 3], [100, 20, 40])
        """
        print(f" Returning {n} points from the circumference of a circle of radius {r}")
        circles = []
        for r, n in zip(r, n):
            t = np.linspace(0, 2* np.pi, n)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = np.zeros(x.size,)
            circles.append(np.c_[x, y, z])
        return np.array(circles).squeeze()
    
    def constant_data(self, constant:float = 0.99):
        print(f"x = {constant}, theta = 0.")
        y = np.linspace(-1., 1., 100)
        test =  np.array([ [constant,x2, 0.] for x2 in y ])
        return test
                    
            
if __name__=='__main__':
    datagen = Datagen()
    a = datagen.random_xyz()
    print(a.shape)
