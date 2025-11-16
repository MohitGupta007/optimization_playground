
import pygmo as pg
import numpy as np
import sys
sys.path.insert(0, '..')
from utils import wrap_pygmo_1d, wrap_pygmo_2d
from visualization_utils import visualize_1d_function_and_evolution, visualize_2d_function_and_evolution
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
run_tracker = RunTracker(csv_path=csv_path)
problem = pg.problem(AckleyProblem(dimension, tracker=run_tracker))
problem_no_tracking = pg.problem(AckleyProblem(dimension))

# Use pygmo's monotonic basin hopping (MBH) algorithm
# MBH wraps a local optimizer (e.g., CMA-ES) and performs basin hopping automatically
algo = pg.algorithm(pg.cmaes(gen=generations))
mbh_algo = pg.algorithm(pg.mbh(algo=None, perturb=1.0, stop=1, seed=42))

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


# Gather all history for visualization
all_x_history, all_f_history = run_tracker.get_history()

print(f"Found {len(unique_minima)} unique minima:")
for idx, minimum in enumerate(unique_minima):
    print(f"Minimum {idx+1}: {minimum}")

# Visualization (optional, similar to ackley_optimization.py)
positions_history = []
for i in range(0, len(all_x_history), pop_size):
    positions = np.array(all_x_history[i:i+pop_size])
    if positions.shape[0] == pop_size:
        positions_history.append(positions)

if dimension == 1:
    func = wrap_pygmo_1d(problem_no_tracking)
    bounds = (-10, 10)
    positions_reshaped = [np.atleast_2d(pos).reshape(-1, 1) for pos in positions_history]
    visualize_1d_function_and_evolution(func, bounds, positions_reshaped, prob_name=problem.get_name(), algo_name=algo.get_name(), interval=interval, unique_minima=unique_minima)
elif dimension == 2:
    func = wrap_pygmo_2d(problem_no_tracking)
    bounds = (-10, 10)
    visualize_2d_function_and_evolution(func, bounds, positions_history, prob_name=problem.get_name(), algo_name=algo.get_name(), interval=interval, unique_minima=unique_minima)
