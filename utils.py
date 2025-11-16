import numpy as np

def wrap_pygmo_1d(func):
    """Returns a vectorized function f(x) from pygmo 1D problem for plotting."""
    def func(x):
        x = np.atleast_1d(x)
        out = np.array([func([xi])[0] for xi in x])
        return out
    return func

def eval_allx_2d(func):
    """Returns a vectorized function f(x,y) from pygmo 2D problem for plotting."""
    def func(x, y):
        x_flat = x.flatten()
        y_flat = y.flatten()
        points = np.column_stack((x_flat, y_flat))
        out = np.array([func(pt)[0] for pt in points])
        return out.reshape(x.shape)
    return func
