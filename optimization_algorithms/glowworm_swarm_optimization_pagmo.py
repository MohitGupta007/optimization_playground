import numpy as np
from scipy.spatial import distance as dist
import pygmo as pg

class GSO:
    """
    Optimized and enhanced Glowworm Swarm Optimization for pygmo.
    """
    def __init__(self, generations=100, step_size=0.2, luciferin_decay=0.1, luciferin_enhance=0.1,
                 beta=1.0, desired_neighbors=5, r0=1.0, rs=3.0, early_stopping_tol=1e-5, random_seed=None):
        self.generations = generations
        self.step_size = step_size
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhance = luciferin_enhance
        self.beta = beta
        self.nt = desired_neighbors
        self.r0 = r0
        self.rs = rs
        self.early_stopping_tol = early_stopping_tol
        if random_seed is not None:
            np.random.seed(random_seed)

    def keep_in_bounds(self, positions, lower_bound, upper_bound):
        return np.clip(positions, lower_bound, upper_bound)

    def evolve(self, pop):
        num_worms = len(pop.get_x())
        dims = len(pop.get_x()[0])
        lb, ub = pop.problem.get_bounds()
        lower_bound = np.array(lb) if hasattr(lb, '__iter__') else np.full(dims, lb)
        upper_bound = np.array(ub) if hasattr(ub, '__iter__') else np.full(dims, ub)

        positions = np.array([pop.get_x()[i][:dims] for i in range(num_worms)])
        sensor_range = np.full(num_worms, self.r0)

        for t in range(self.generations):
            if t == 0:
                luciferin = np.array([pop.get_f()[i][0] for i in range(num_worms)])
            else:
                fitness = np.array([pop.problem.fitness(pos)[0] for pos in positions])
                luciferin = (1 - self.luciferin_decay) * luciferin + self.luciferin_enhance * (-fitness)

            dist_matrix = dist.cdist(positions, positions)
            prev_positions = positions.copy()

            for i in range(num_worms):
                # Vectorized neighbor selection for efficiency
                # Mask of valid neighbors (distance AND luciferin condition)
                mask = (dist_matrix[i] < sensor_range[i]) & (luciferin[i] < luciferin)
                mask[i] = False
                neighbors = np.where(mask)[0]

                if neighbors.size > 0:
                    dL = luciferin[neighbors] - luciferin[i]
                    total = dL.sum()
                    if total > 0:
                        pij = dL / total
                    else:
                        pij = np.ones_like(dL) / len(dL)

                    j = np.random.choice(neighbors, p=pij)
                    diff = positions[j] - positions[i]
                    norm = np.linalg.norm(diff)
                    direction = diff / (norm + 1e-12) if norm > 0 else 0
                    positions[i] += self.step_size * direction

                    # Boundaries
                    positions[i] = self.keep_in_bounds(positions[i], lower_bound, upper_bound)

                    # Sensor update (core logic)
                    sensor_range[i] = min(self.rs, max(0, sensor_range[i] + self.beta * (self.nt - len(neighbors))))
                else:
                    all_distances = dist_matrix[i][np.arange(num_worms) != i]
                    if all_distances.size > 0:
                        nearest = np.min(all_distances)
                        sensor_range[i] = nearest + 1e-1
                                
            # Uncomment to trace evolution:
            # print(f"Generation {t}: Fitness best = {fitness.min():.4f}, mean = {fitness.mean():.4f}")

            #step size decay
            self.step_size *= 0.99

            # Early stopping: terminate if movement is minimal
            # if np.max(np.linalg.norm(positions - prev_positions, axis=1)) < self.early_stopping_tol:
            #     break

        for i in range(num_worms):
            individual = pop.get_x()[i].copy()
            individual[:dims] = positions[i]
            pop.set_x(i, individual)
        return pop