import numpy as np


def gen_data(n, a=3.0, b=1.0):
    x = np.linspace(-5, 5, n)
    epsilon = np.random.normal(0, 0.1, n)
    y = a * x + b + epsilon

    return x.reshape(n, 1), y.reshape(n, 1)
