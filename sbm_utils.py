from scipy.special import expit as logistic
from scipy.stats import norm
import numpy as np


def generate_clusters(n=100, pi_vector=None):
    if pi_vector is None:
        pi_vector = [0.3, 0.4, 0.3]
    clusters_splits = list((np.round(n * np.cumsum(pi_vector)).astype('int')))
    k = len(pi_vector)
    c = np.zeros(n)
    for k_ind in range(k - 1):
        c[clusters_splits[k_ind]:clusters_splits[k_ind + 1]] = k_ind + 1
    return c


def generate_sbm_adj(n, pi_vector, theta_in, theta_out):
    # type: (int, list, float, float) -> object
    c = generate_clusters(n, pi_vector)
    Adj = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if c[i] == c[j]:
                Adj[i, j] = int(np.random.randint(100) < 100 * theta_in)
                Adj[j, i] = Adj[i, j]
            else:
                Adj[i, j] = int(np.random.randint(100) < 100 * theta_out)
                Adj[j, i] = Adj[i, j]
    return Adj


def generate_wsbm_adj(n, pi_vector, theta_in=3, theta_out=-3):
    c = generate_clusters(n, pi_vector)
    Adj = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if c[i] == c[j]:
                Adj[i, j] = logistic(norm.rvs(theta_in))
                Adj[j, i] = Adj[i, j]
            else:
                Adj[i, j] = logistic(norm.rvs(theta_out))
                Adj[j, i] = Adj[i, j]
    np.fill_diagonal(Adj, 1)
    return Adj


def random_permute(matrix):
    n = matrix.shape[0]
    order = np.random.permutation(n)
    permuted = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            permuted[i, j] = matrix[order[i], order[j]]
            permuted[j, i] = matrix[order[j], order[i]]
    return order, permuted


def inverse_permute(order, permuted):
    n = permuted.shape[0]
    new_order = np.argsort(order)
    inverse = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            inverse[i, j] = permuted[new_order[i], new_order[j]]
            inverse[j, i] = permuted[new_order[j], new_order[i]]
    return inverse
