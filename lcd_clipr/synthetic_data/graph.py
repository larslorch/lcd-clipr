"""
Taken from
https://github.com/larslorch/avici/blob/main/avici/synthetic/graph.py
"""

import igraph as ig
import numpy as onp
import random as pyrandom


def graph_to_mat(g):
    """Returns adjacency matrix of ig.Graph object """
    return onp.array(g.get_adjacency().data).astype(int)


class ErdosRenyi:
    """
    Erdos-Renyi random graph

    Args:
        edges_per_var (float): expected number of edges per node, scaled to account for number of nodes
    """

    def __init__(self, edges_per_var):
        self.edges_per_var = edges_per_var

    def __call__(self, rng, n_vars):
        # select p s.t. we get requested edges_per_var in expectation
        n_edges = self.edges_per_var * n_vars
        p = min(n_edges / (n_vars * (n_vars - 1)), 0.99)
        mat = rng.binomial(n=1, p=p, size=(n_vars, n_vars)).astype(int)

        # randomly permute
        p = rng.permutation(onp.eye(n_vars).astype(int))
        mat = p.T @ mat @ p
        return mat


class ScaleFree:
    """
    Barabasi-Albert (scale-free)
    Power-law in-degree (few nodes have high in-degree)

    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process.
            Higher values make few nodes have high in-degree.

    """
    def __init__(self, edges_per_var, power=1.0, p_flip=0.1):
        self.edges_per_var = edges_per_var
        self.power = power
        self.p_flip = p_flip

    def __call__(self, rng, n_vars):
        pyrandom.seed(rng.bit_generator.state["state"]["state"])  # seed pyrandom based on state of numpy rng
        _ = rng.normal()  # advance rng state by 1
        perm = rng.permutation(n_vars).tolist()
        g = ig.Graph.Barabasi(n=n_vars, m=self.edges_per_var, directed=True, power=self.power).permute_vertices(perm)
        mat = graph_to_mat(g)
        flip = rng.binomial(n=1, p=self.p_flip, size=(n_vars, n_vars)).astype(int)
        mat = mat * (1 - flip) + mat.T * flip
        return mat


class ScaleFreeOutgoing(ScaleFree):
    """
    Barabasi-Albert (scale-free)
    Power-law out-degree (few nodes have high out-degree)

    Args:
        edges_per_var (int): number of edges per node
        power (float): power in preferential attachment process.
            Higher values make few nodes have high out-degree.

    """
    def __init__(self, edges_per_var, power=1, p_flip=0.1):
        super().__init__(edges_per_var=edges_per_var, power=power, p_flip=p_flip)

    def __call__(self, rng, n_vars):
        mat = super().__call__(rng, n_vars)
        return mat.T

