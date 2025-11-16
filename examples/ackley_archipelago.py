import pygmo as pg
import numpy as np

# Problem dimension
dimension = 2  # adjust as needed
tolerance = 1e-1  # tolerance for uniqueness check

# Total evaluation budget (approximate)
total_budget = 100000

# Number of islands in the archipelago
num_islands = 100

# Evaluations per island (divides total budget roughly equally)
evals_per_island = total_budget // num_islands

# Create the problem instance
problem = pg.problem(pg.rosenbrock(dimension))

# Define different algorithms for diversity
algos = [
    pg.algorithm(pg.nspso(gen=evals_per_island))
]

# Create the archipelago
archi = pg.archipelago()

# Add islands to archipelago with different algorithms and seeds
for i in range(num_islands):
    algo = algos[i % len(algos)]
    pop = pg.population(problem, size=30, seed=i)
    isl = pg.island(algo=algo, pop=pop)  # Use keyword arguments
    archi.push_back(isl)

# Evolve the archipelago in parallel with migration rounds
num_migrations = 10
for _ in range(num_migrations):
    archi.evolve()
    archi.wait()

# Collect all solutions from all islands
all_solutions = []


def is_unique(minima_list, point):
    for p in minima_list:
        if np.linalg.norm(np.array(p) - np.array(point)) < tolerance:
            return False
    return True

for isl in archi:
    champ_x = isl.get_population().champion_x
    if is_unique(all_solutions, champ_x):
        all_solutions.append(champ_x)

print(f"Found {len(all_solutions)} unique local minima:")

for idx, minimum in enumerate(all_solutions):
    print(f"Minimum {idx + 1}: {minimum}")
