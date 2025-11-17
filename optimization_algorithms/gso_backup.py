import numpy as np
from scipy.spatial import distance as dist
import copy
import pygmo as pg


class GSO:
    def __init__(self, gen=100, influence_factor=10, max_jitter=0.1,
                 luciferin_decay=0.4, luciferin_enhance=0.6, max_step_norm=0.1):
        self.nturns = gen
        self.influence_factor = influence_factor
        self.max_jitter = max_jitter
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhance = luciferin_enhance
        self.max_step_norm = max_step_norm
        self.luciferin = None  # Will initialize on first evolve call

    def keep_in_bounds(self, positions, lower_bound, upper_bound):
        return np.clip(positions, lower_bound, upper_bound)

    def evolve(self, pop):
        num_worms = len(pop.get_x())
        dims = len(pop.get_x()[0])
        lb, ub = pop.problem.get_bounds()

        lower_bound = np.array(lb) if hasattr(lb, '__iter__') else np.full(dims, lb)
        upper_bound = np.array(ub) if hasattr(ub, '__iter__') else np.full(dims, ub)

        positions = np.array([pop.get_x()[i][:dims] for i in range(num_worms)])

        # Initialize luciferin values with fitness scaled + offset
        fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
        min_lb = np.min(lower_bound)
        luciferin = (fitness_vals + min_lb) / self.influence_factor if self.luciferin is None else self.luciferin.copy()

        for _ in range(self.nturns):
            # Update luciferin with decay and enhancement from current fitness
            # fitness_vals = np.array([pop.problem.fitness(pos)[0] for pos in positions])
            # luciferin = (1 - self.luciferin_decay) * luciferin + self.luciferin_enhance * (fitness_vals + min_lb) / self.influence_factor
            # Higher luciferin at lower fitness should attract glowworms
            raw_scores = np.array([pop.problem.fitness(pos)[0] for pos in positions])
            max_fit = np.max(raw_scores)
            min_fit = np.min(raw_scores)
            # Invert and scale luciferin:  higher luciferin means better (lower) fitness
            luciferin = (max_fit - raw_scores) / self.influence_factor

            # Influence matrix: pairwise distances where dist <= luciferin_j
            dist_matrix = dist.cdist(positions, positions)
            mask = dist_matrix <= luciferin[np.newaxis, :]
            np.fill_diagonal(mask, False)
            influence_mat = np.where(mask, dist_matrix, 0)

            # Movement

            # Condition: influencers have higher luciferin than self and distance in influence range
            cond = luciferin[:, np.newaxis] < luciferin[np.newaxis, :]
            valid_moves = (influence_mat != 0) & cond

            percent_move = np.zeros_like(influence_mat)
            divisor = np.broadcast_to(luciferin, influence_mat.shape)
            percent_move[valid_moves] = 1 - (influence_mat[valid_moves] / divisor[valid_moves])

            diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
            weighted_move = diff * percent_move[:, :, np.newaxis] / 20.0  # smaller step divisor for stability

            raw_move = weighted_move.sum(axis=1)

            # Clip step norm per worm to max_step_norm
            norms = np.linalg.norm(raw_move, axis=1)
            scaling = np.minimum(1, self.max_step_norm / (norms + 1e-10))
            total_move = raw_move * scaling[:, np.newaxis]

            jitter = self.max_jitter * (np.random.rand(num_worms, dims) * 2 - 1)

            new_positions = positions + total_move + jitter
            positions = self.keep_in_bounds(new_positions, lower_bound, upper_bound)

        # Update population in pygmo with new positions and fitness
        for i in range(num_worms):
            individual = pop.get_x()[i].copy()
            individual[:dims] = positions[i]
            pop.set_x(i, individual)
            # fitness auto-updates; no need to set manually

        self.luciferin = luciferin  # save luciferin state for next evolve if used multiple times
        return pop