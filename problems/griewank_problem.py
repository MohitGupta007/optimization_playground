import pygmo as pg
import numpy as np

class GriewankProblem:
    def __init__(self, dim, tracker=None):
        self.dim = dim
        self.lb = [-10.0] * dim
        self.ub = [10.0] * dim 
        self.ackley = pg.problem(pg.griewank(dim))
        self.run_tracker = tracker
    
    def fitness(self, x):
        f = self.ackley.fitness(x)
        
        if self.run_tracker is not None:
            self.run_tracker.log_evaluation(np.array(x), f[0])
        
        return f
    
    def get_bounds(self):
        return (self.lb, self.ub)
    
    def get_nobj(self):
        return 1
    
    def get_nix(self):
        return 0
    
    def get_name(self):
        return f"Ackley problem in {self.dim} dimensions"