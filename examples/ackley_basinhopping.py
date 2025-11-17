
import pygmo as pg
import numpy as np
import sys
sys.path.insert(0, '..')
from visualization_utils import visualize_optimization
from problems.ackley_problem import AckleyProblem
from run_tracker import RunTracker

# Problem settings
dimension = 1  # Change as needed
interval = 100
pop_size = 100
generations = 200
num_restarts = 30
csv_path = '/home/mohit/projects/optimization_playground/runs/ackley_basinhopping_log.csv'

# Set up tracker and problem
run_tracker = RunTracker(csv_path=csv_path, pop_size=pop_size, generations=generations)
# Create a tracked problem instance and a separate no-tracking problem for visualization
problem_instance = AckleyProblem(dimension, tracker=run_tracker)
problem = pg.problem(problem_instance)
problem_no_tracking = pg.problem(AckleyProblem(dimension))

# Use pygmo's monotonic basin hopping (MBH) algorithm
# MBH wraps a local optimizer (e.g., CMA-ES) and performs basin hopping automatically
algo = pg.algorithm(pg.cmaes(gen=generations))
mbh_algo = pg.algorithm(pg.mbh(algo=None, perturb=1.0, stop=1, seed=42))

# Set tracker metadata for visualization
run_tracker.set_algo_name(mbh_algo.get_name())
run_tracker.set_problem_name(problem.get_name())
run_tracker.set_bounds(problem_no_tracking.get_bounds())

# Basin-hopping logic
unique_minima = []
tolerance = 1e-1

def is_unique(minima_list, point):
    for p in minima_list:
        if np.linalg.norm(np.array(p) - np.array(point)) < tolerance:
            return False
    return True



# Use MBH to perform basin hopping
pop = pg.population(problem, size=pop_size)
pop = mbh_algo.evolve(pop)

# Collect unique minima from the final population
unique_minima = []
for i in range(pop_size):
    candidate = pop.get_x()[i]
    if is_unique(unique_minima, candidate):
        unique_minima.append(candidate)

print(f"Found {len(unique_minima)} unique minima:")
for idx, minimum in enumerate(unique_minima):
    print(f"Minimum {idx+1}: {minimum}")

# Visualization
problem_instance.run_tracker = None
func = problem_no_tracking.fitness
visualize_optimization(func, run_tracker, dimension, interval=interval, unique_minima=unique_minima)
