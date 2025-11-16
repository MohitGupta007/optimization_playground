import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def visualize_optimization(func, run_tracker, dimension, interval=100, blit=False, show=True, unique_minima=None):
    if(run_tracker is None):
        raise ValueError("[Visualization] Invalid Run Tracker")
    if(dimension not in [1,2]):
        raise ValueError("[Visualization] Dimension must be 1 or 2")
    if(func is None):
        raise ValueError("[Visualization] Function not provided")

    
    all_x_history, all_f_history = run_tracker.get_history()
    pop_size = run_tracker.pop_size

    positions_history = []
    for i in range(0, len(all_x_history), pop_size):
        positions = all_x_history[i:i+pop_size,:]
        positions_history.append(positions)

    if dimension == 1:
        positions_reshaped = [np.atleast_2d(pos).reshape(-1, 1) for pos in positions_history]
        visualize_1d_function_and_evolution(func, run_tracker.bounds, positions_reshaped, prob_name=run_tracker.problem_name, algo_name=run_tracker.algo_name, interval=interval, blit=blit, show=show, unique_minima=unique_minima)
    elif dimension == 2:
        visualize_2d_function_and_evolution(func, run_tracker.bounds, positions_history, prob_name=run_tracker.problem_name, algo_name=run_tracker.algo_name, interval=interval, blit=blit, show=show, unique_minima=unique_minima)

def visualize_1d_function_and_evolution(func, bounds, positions, prob_name="", algo_name="", interval=200, blit=False, show=True, unique_minima=None):
    """
    Visualizes a 1D function and animates population positions over generations.

    Parameters:
    - func: callable, function f(x) to visualize
    - bounds: tuple (lower, upper) domain bounds for plotting
    - positions: list of np.ndarray, population positions per generation (shape: (pop_size, 1))
    - prob_name: string, name of the problem being solved
    - algo_name: string, name of the algorithm used
    - interval: int, animation frame interval in milliseconds
    - unique_minima: list of np.ndarray, coordinates of unique minima to mark
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    x_vals = np.linspace(bounds[0], bounds[1], 500)
    y_vals = np.array([func(x) for x in x_vals])
    ax.plot(x_vals, y_vals, label='Objective function')
    scatter_pop = ax.scatter([], [], color='red', label='Population samples')
    
    # Mark unique minima
    if unique_minima is not None:
        for idx, minimum in enumerate(unique_minima):
            x_min = np.atleast_1d(minimum).flatten()[0]
            y_min = func(x_min)
            label = 'Unique minima' if idx == 0 else ''
            ax.plot(x_min, y_min, 'g*', markersize=15, label=label)
    ax.set_xlim(bounds)
    ax.set_ylim(np.min(y_vals)*1.1, np.max(y_vals)*1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()

    def init():
        scatter_pop.set_offsets(np.empty((0, 2)))
        return scatter_pop,

    def update(frame):
        i, pos = frame
        x_pop = np.atleast_1d(pos).flatten()
        y_pop = np.array([func(x) for x in pos])
        scatter_pop.set_offsets(np.c_[x_pop, y_pop])
        ax.set_title(f"{prob_name}, {algo_name} - Generation {i}")
        return scatter_pop,

    frames = list(enumerate(positions))
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval, blit=blit)
    if show:
        plt.show()
    return anim


def visualize_2d_function_and_evolution(func, bounds, positions, prob_name="", algo_name="", interval=200, levels=50, cmap='viridis', blit=False, show=True, unique_minima=None):
    """
    Visualizes a 2D function contour and animates population positions over generations.

    Parameters:
    - func: callable, function f(x, y) to visualize over meshgrid
    - bounds: tuple (lower, upper) domain bounds for both x and y
    - positions: list of np.ndarray, population positions per generation (shape: (pop_size, 2))
    - title: string, plot title prefix
    - interval: int, animation frame interval in milliseconds
    - levels: int, number of contour levels
    - cmap: string, colormap name
    - unique_minima: list of np.ndarray, coordinates of unique minima to mark
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    x_vals = np.linspace(bounds[0], bounds[1], 100)
    y_vals = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.array([func([x, y]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    fig.colorbar(contour)
    scatter_pop = ax.scatter([], [], color='red', label='Population samples')
    
    # Mark unique minima
    if unique_minima is not None:
        x_minima = [np.atleast_1d(m)[0] for m in unique_minima]
        y_minima = [np.atleast_1d(m)[1] for m in unique_minima]
        ax.scatter(x_minima, y_minima, color='green', marker='*', s=500, label='Unique minima', edgecolors='black', linewidths=1.5)
    ax.set_xlim(bounds[:,0])
    ax.set_ylim(bounds[:,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    def init():
        scatter_pop.set_offsets(np.empty((0, 2)))
        return scatter_pop,

    def update(frame):
        i, pos = frame
        x_pop = np.atleast_1d(pos)[:, 0]
        y_pop = np.atleast_1d(pos)[:, 1]
        scatter_pop.set_offsets(np.c_[x_pop, y_pop])
        ax.set_title(f"{prob_name}, {algo_name} - Generation {i}")
        return scatter_pop,

    frames = list(enumerate(positions))
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval, blit=blit)
    if show:
        plt.show()
    return anim
