import numpy as np
from scipy.spatial import distance as dist
import copy
import pygmo as pg


class GSO:
    def __init__(self, generations=100, influence_factor=20, 
                 max_jitter=0.1, min_jitter=0.001,
                 luciferin_decay=0.1, luciferin_enhance=0.1,
                 max_step_norm=0.1,
                 max_influence_radius=5.0, min_influence_radius=0.5):
        self.nturns = generations
        self.influence_factor = influence_factor
        self.initial_jitter = max_jitter
        self.min_jitter = min_jitter
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhance = luciferin_enhance
        self.max_step_norm = max_step_norm
        self.initial_influence_radius = max_influence_radius
        self.min_influence_radius = min_influence_radius
        self.luciferin = None

    def keep_in_bounds(self, positions, lower_bound, upper_bound):
        return np.clip(positions, lower_bound, upper_bound)

    def _decay(self, start, end, current_iter):
        # Linear decay from start to end over nturns iterations
        return max(end, start - (start - end) * (current_iter / self.nturns))

    def evolve(self, pop):
        num_worms = len(pop.get_x())
        dims = len(pop.get_x()[0])
        lb, ub = pop.problem.get_bounds()

        lower_bound = np.array(lb) if hasattr(lb, '__iter__') else np.full(dims, lb)
        upper_bound = np.array(ub) if hasattr(ub, '__iter__') else np.full(dims, ub)

        positions = np.array([pop.get_x()[i][:dims] for i in range(num_worms)])

        # Initialize luciferin inverted so best solution has highest luciferin
        fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
        max_fit = np.max(fitness_vals)
        luciferin = (max_fit - fitness_vals) / self.influence_factor if self.luciferin is None else self.luciferin.copy()

        for it in range(self.nturns):
            # Decay jitter and influence radius during optimization
            current_jitter = self._decay(self.initial_jitter, self.min_jitter, it)
            current_influence_radius = self._decay(self.initial_influence_radius, self.min_influence_radius, it)

            # Update luciferin values
            fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
            max_fit = np.max(fitness_vals)
            luciferin = (1 - self.luciferin_decay) * luciferin + \
                        self.luciferin_enhance * ((max_fit - fitness_vals) / self.influence_factor)

            # Compute pairwise distances
            dist_matrix = dist.cdist(positions, positions)

            # Influence mask uses current decayed influence radius and luciferin
            mask = (dist_matrix <= luciferin[np.newaxis, :]) & (dist_matrix <= current_influence_radius)
            np.fill_diagonal(mask, False)

            influence_mat = np.where(mask, dist_matrix, 0)

            cond = luciferin[:, np.newaxis] < luciferin[np.newaxis, :]
            valid_moves = (influence_mat != 0) & cond

            percent_move = np.zeros_like(influence_mat)
            divisor = np.broadcast_to(luciferin, influence_mat.shape)
            percent_move[valid_moves] = 1 - (influence_mat[valid_moves] / divisor[valid_moves])

            diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
            weighted_move = diff * percent_move[:, :, np.newaxis] / 20.0

            raw_move = weighted_move.sum(axis=1)

            norms = np.linalg.norm(raw_move, axis=1)
            scaling = np.minimum(1, self.max_step_norm / (norms + 1e-10))
            total_move = raw_move * scaling[:, np.newaxis]

            jitter = current_jitter * (np.random.rand(num_worms, dims) * 2 - 1)

            new_positions = positions + total_move + jitter
            positions = self.keep_in_bounds(new_positions, lower_bound, upper_bound)

        for i in range(num_worms):
            individual = pop.get_x()[i].copy()
            individual[:dims] = positions[i]
            pop.set_x(i, individual)

        self.luciferin = luciferin
        return pop