import numpy as np

# Define a high-dimensional problem by subclassing pygmo.problem.base
class SphereProblem:
    def __init__(self, dim):
        self.dim = dim
        self.lb = [-5.0] * dim  # lower bounds
        self.ub = [5.0] * dim   # upper bounds
    
    def fitness(self, x):
        # Sphere function: minimize sum of squares
        return [sum(xi*xi for xi in x)]
    
    def get_bounds(self):
        return (self.lb, self.ub)
    
    def get_nobj(self):
        return 1
    
    def get_nix(self):
        return 0
    
    def get_name(self):
        return f"Sphere problem in {self.dim} dimensions"