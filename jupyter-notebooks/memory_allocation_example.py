
import numpy as np

def sum_of_square_vectorized(X):
    x = np.asarray(X)
    x_sq = np.square(x) # x ** 2
    res = x_sq.sum()
    return res


def sum_of_square_for_loop(X):
    res = 0
    for x in X:
        res += x ** 2
    return res
