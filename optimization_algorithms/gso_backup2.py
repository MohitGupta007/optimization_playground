import numpy as np
from scipy.spatial import distance as dist
import copy
import pygmo as pg


class GSO:
    def __init__(self, gen=100, influence_factor=5, max_jitter=0.01,
                 luciferin_decay=0.3, luciferin_enhance=0.8, max_step_norm=0.5,
                 max_influence_radius=1.0):
        self.nturns = gen
        self.influence_factor = influence_factor
        self.max_jitter = max_jitter
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhance = luciferin_enhance
        self.max_step_norm = max_step_norm
        self.max_influence_radius = max_influence_radius
        self.luciferin = None

    def keep_in_bounds(self, positions, lower_bound, upper_bound):
        return np.clip(positions, lower_bound, upper_bound)

    def evolve(self, pop):
        num_worms = len(pop.get_x())
        dims = len(pop.get_x()[0])
        lb, ub = pop.problem.get_bounds()

        lower_bound = np.array(lb) if hasattr(lb, '__iter__') else np.full(dims, lb)
        upper_bound = np.array(ub) if hasattr(ub, '__iter__') else np.full(dims, ub)

        positions = np.array([pop.get_x()[i][:dims] for i in range(num_worms)])

        # Compute fitness and initialize luciferin with inversion so best fitness has highest luciferin
        fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
        max_fit = np.max(fitness_vals)
        min_fit = np.min(fitness_vals)
        luciferin = (max_fit - fitness_vals) / self.influence_factor if self.luciferin is None else self.luciferin.copy()

        for _ in range(self.nturns):
            # Update fitness and luciferin with decay and enhancement
            fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
            max_fit = np.max(fitness_vals)
            luciferin = (1 - self.luciferin_decay) * luciferin + \
                        self.luciferin_enhance * ((max_fit - fitness_vals) / self.influence_factor)

            # Pairwise distances
            dist_matrix = dist.cdist(positions, positions)
            # Mask for neighbors with distance <= luciferin of j AND <= max_influence_radius
            mask = (dist_matrix <= luciferin[np.newaxis, :]) & (dist_matrix <= self.max_influence_radius)
            np.fill_diagonal(mask, False)

            influence_mat = np.where(mask, dist_matrix, 0)

            cond = luciferin[:, np.newaxis] < luciferin[np.newaxis, :]
            valid_moves = (influence_mat != 0) & cond

            percent_move = np.zeros_like(influence_mat)
            divisor = np.broadcast_to(luciferin, influence_mat.shape)
            percent_move[valid_moves] = 1 - (influence_mat[valid_moves] / divisor[valid_moves])

            diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # shape (num_worms, num_worms, dims)
            weighted_move = diff * percent_move[:, :, np.newaxis] / 20.0

            raw_move = weighted_move.sum(axis=1)

            # Clip step norm to max_step_norm per glowworm
            norms = np.linalg.norm(raw_move, axis=1)
            scaling = np.minimum(1, self.max_step_norm / (norms + 1e-10))
            total_move = raw_move * scaling[:, np.newaxis]

            jitter = self.max_jitter * (np.random.rand(num_worms, dims) * 2 - 1)

            new_positions = positions + total_move + jitter
            positions = self.keep_in_bounds(new_positions, lower_bound, upper_bound)

        # Update pygmo population with new positions; fitness updates automatically
        for i in range(num_worms):
            individual = pop.get_x()[i].copy()
            individual[:dims] = positions[i]
            pop.set_x(i, individual)

        self.luciferin = luciferin
        return pop