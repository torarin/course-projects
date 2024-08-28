import numpy as np


def mesh_function(f, t):
    out = np.empty(t.size)
    for i in range(t.size):
        out[i] = f(t[i])
    return out


def func(t):
    if t <= 3:
        return np.exp(-t)
    return np.exp(-3*t)

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
