__author__ = 'gru'

from collections import namedtuple

import numpy as np
from scipy import sparse as sps


LogMapKey = namedtuple("LogMapKey", "mu x0 m")

# Defines the range of the logistic map parameter \mu
MU_RANGE = (3.569945, 4.0)
# Defines the range of the logistic map parameter m
M_BOUNDS = (0xF, 0xFFFF)


def log_map(x, mu):
    """Logistic map equation. Takes x_k and mu and calculates x_{k+1}

    :param x: the current value
    :param mu: 0<\mu<4; to get complex dynamic behaviour choose 3.569945..<mu<4
    :return: x_{k+1}
    """
    return mu * x * (1 - x)


def generate_permutation(size, mu=3.569945, x0=0.5, m=100):
    """Generates a permutation with the help of a logistic map.
    To get a sequence of pseudorandom numbers the logistic map
    equation is repeatedly applied to a start value x0. The list of random
    numbers is then sorted, which forms the permutation.

    :param size: the size of the permutation
    :param mu: 0<\mu<4 parameter of the logistic map
    :param x0: the start value between 0...1
    :param m: 'lead time' - after m iterations the resulting numbers are taken
    into account
    :return: permutation: the resulting permutation
    """
    x_i = x0
    n = m + size
    results = np.zeros(size)
    i = 0
    while i < n:
        x_i = log_map(x_i, mu)

        if i >= m:
            results[i - m] = x_i

        i += 1
    permutation = results.argsort(kind='mergesort')

    return permutation


def generate_random_permutation(size):
    """Generates a random permutation by choosing random parameters for the
    logistic map, which is then used to build a chaotic sequence,
    whose sorting forms the permutation. It is checked, that the generated
    permutation equals not the identity mapping.

    :param size:the length of the permutation
    :return: the pseudo-random permutation
    """
    perm = identity(size)
    while np.array_equal(perm, identity(size)):
        # Generate logistic map parameters
        map_key = LogMapKey(mu=random_mu(),
                            x0=np.random.random_sample(),
                            m=np.random.randint(M_BOUNDS[0],
                                                M_BOUNDS[1]))

        # Generate rho
        perm = generate_permutation(size, *map_key)

    return perm


def random_mu():
        """Generates a pseudo-random \mu, which serves as one parameter in
        the logistic map. Prover.MU_RANGE[0]<\mu<Prover.MU_RANGE[1]

        :return: mu: pseudo-random generated \mu
        """
        return (MU_RANGE[1] - MU_RANGE[
            0]) * np.random.random_sample() + MU_RANGE[0]


def identity(size):
    """Generates an identity permutation - trivial.

    :param size: size of the permutation
    :return: the identity permutation
    """
    return np.arange(0, size, 1)


def invert(permutation):
    """Inverts a given permutation

    :param permutation: the permutation to be inverted
    :return: inv_perm: the inverted permutation
    """
    inv_perm = np.zeros(len(permutation), dtype=permutation[0].dtype)

    for i, v in enumerate(permutation):
        inv_perm[v] = i

    return inv_perm


def compose_permutations(p2, p1):
    """Calculates p2 \circ p1 and returns the resulting permutation.

    :param p1: a permutation
    :param p2: another permutation
    :return: p3: the resulting permutation p2 \circ p1
    """
    p3 = np.zeros_like(p1)

    for i, v in enumerate(p1):
        p3[i] = p2[v]

    return p3


def permute_list(l, permutation):
    """Permutes a list according to the given permutation

    :param l: the list to permute
    :param permutation: the permutation to apply
    :return: l2: the permuted list
    """
    l1 = l
    l2 = np.zeros_like(l1)

    for i, v in enumerate(permutation):
        l2[v] = l1[i]

    return l2


def permute_graph(graph, permutation):
    """Shuffles the nodes of a given graph according to a given permutation
    and forms the resulting adjacency matrix.

    :param graph: a sparse adjacency matrix (in csr format), which represents
    the graph
    :param permutation: the permutation to apply
    :return: shuffled_graph: the resulting graph's adjacency matrix in
    (csr format)
    """
    nnz_i, nnz_j = graph.nonzero()
    graph_adjacency = graph.toarray()
    shuffled_adjacency = np.zeros(
        (graph_adjacency.shape[0], graph_adjacency.shape[1]))

    for x in range(len(nnz_i)):
        new_i = permutation[nnz_i[x]]
        new_j = permutation[nnz_j[x]]

        shuffled_adjacency[new_i][new_j] = graph_adjacency[nnz_i[x]][nnz_j[x]]

    shuffled_graph = sps.csr_matrix(shuffled_adjacency)
    return shuffled_graph


def permute_histogram(samples, bins, permutation):
    """Permutes the histogram bins according to the given permutation.

    :param samples: the samples to be modified
    :param bins: a list, that defines the bin edges
    :param permutation: the permutation to apply
    :return: shuffled_samples: the modified samples
    """
    bin_width = abs(bins[1] - bins[0])

    # make a copy, shuffling cannot be performed in situ
    shuffled_samples = np.empty_like(samples)
    shuffled_samples[:] = samples

    for id1, v in enumerate(permutation):

        # retrieve index of target bin
        id2 = v

        if id1 < len(bins) - 1:
            for i, x in enumerate(samples):
                if bins[id1] <= x < bins[id1 + 1]:
                    shuffled_samples[i] = x + ((id2 - id1) * bin_width)

    return shuffled_samples


def permute_wmk_bins(bin_pairs, permutation):
    """Maps the bin ids in the detection key (consists of bin pairs)
    according to the given permutation.

    :param bin_pairs: the list of bin ids specifying the pairs of bins
    :param permutation: the permutation to apply
    :return: shuffled_bin_pairs: the watermark key with the substituted bin ids
    """
    shuffled_bin_pairs = np.empty_like(bin_pairs)

    for i, pair in enumerate(bin_pairs):
        for j, b in enumerate(pair):
            shuffled_bin_pairs[i][j] = permutation[b]

    return shuffled_bin_pairs