import pygmo as pg
import sys
sys.path.insert(0, '..')

from visualization_utils import visualize_optimization
from problems.ackley_problem import AckleyProblem
from problems.rosenbrock_problem import RosenbrockProblem
import numpy as np
from run_tracker import RunTracker

# Problem dimension
dimension = 1
interval = 100
pop_size = 10
generations = 500
csv_path = '/home/mohit/projects/optimization_playground/runs/ackley_optimization_log.csv'

# Create a custom Ackley problem instance
run_tracker = RunTracker(csv_path=csv_path, pop_size=pop_size, generations=generations)
problem = AckleyProblem(dimension, tracker=run_tracker)
# problem = RosenbrockProblem(dimension, tracker=run_tracker)
pagmo_problem = pg.problem(problem)

# Choose an algorithm to solve the problem
algo = pg.algorithm(pg.cmaes(gen=generations))

run_tracker.set_algo_name(algo.get_name())
run_tracker.set_problem_name(pagmo_problem.get_name())
run_tracker.set_bounds(pagmo_problem.get_bounds())

# Create a population of candidate solutions
pop = pg.population(pagmo_problem, size=pop_size)

# algo.set_verbosity(1)

# Run the evolution
pop = algo.evolve(pop)

# Get the best solution found
best_x = pop.champion_x
best_f = pop.champion_f

print(f"Best solution found:\n {best_x}")
print(f"Objective value:\n {best_f}")

# Visualization
problem.run_tracker = None  # Disable tracking for visualization
func = pg.problem(problem).fitness
visualize_optimization(func, run_tracker, dimension, interval=interval)
