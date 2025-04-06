import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import networkx as nx
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
import copy
import os

from brainpy import share
from brainpy.connect import TwoEndConnector
from brainpy.types import Shape, ArrayType
from brainpy.check import is_initializer
from brainpy import odeint, sdeint, JointEq
from brainpy._src.connect.base import get_idx_type
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial


def pytree_to_numpy(pytree):
    """
    Recursively traverse `pytree`, converting JAX arrays (jnp.ndarray)
    to NumPy arrays (np.ndarray).
    """

    def convert_if_jax_array(x):
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
        else:
            return x

    return jax.tree_map(convert_if_jax_array, pytree)


def format_input(inputs):
    # Convert all elements to Tuples of (string, jnparrays)
    return [(str(key), jnp.asarray(value)) for key, value in inputs]


def remove_key(d, key):
    if not isinstance(d, dict):
        return d

    cleaned_dict = {}
    for k, value in d.items():
        if k == key:
            continue  # Skip this key
        elif isinstance(value, dict):
            cleaned_dict[k] = remove_key(value, key)
        else:
            cleaned_dict[k] = value

    return cleaned_dict


# Wrapper that extracts the parameters from the proj object
def indegrees(proj):  # Not jax-compatible
    _bincount = jax.jit(jnp.bincount, static_argnames=["length"])
    post_size = proj.post.size  # Should be a tuple of ints
    N = int(jnp.prod(jnp.array(post_size)))
    indices = jnp.asarray(proj.comm.indices, dtype=jnp.int32)
    return _bincount(indices, length=N)


def indegrees_static(indices, N):  # Jax-compatible if N is static
    _bincount = jax.jit(jnp.bincount, static_argnames=["length"])
    indices = jnp.asarray(indices, dtype=jnp.int32)
    return _bincount(indices, length=np.prod(N))


def indegree(proj):
    N = int(jnp.prod(jnp.array(proj.post.size)))
    indices = jnp.array(proj.comm.indices, dtype=int)
    in_degree = jnp.bincount(indices, length=N)
    return jnp.mean(in_degree)


def correlate_weights(proj, J, N, key):
    k = indegrees_static(proj.comm.indices, N)

    nonzero_mask = k > 0
    k_nonzero = jnp.where(nonzero_mask, k, 0)
    sqrtk_nonzero = jnp.sqrt(
        jnp.where(nonzero_mask, k, 1)
    )  # sqrt(k[i]) if k>0, else sqrt(1)=1 to avoid nans

    sum_k = jnp.sum(k_nonzero)
    sum_sqrt = jnp.sum(sqrtk_nonzero)
    # If *all* k are zero, sum_sqrt=0 => we define J_rec=0
    J_rec = jnp.where(sum_sqrt > 0, J * sum_k / sum_sqrt, 0.0)

    indices = proj.comm.indices
    ws_orig = proj.comm.weight
    k_safe = jnp.where(k > 0, k, jnp.inf)

    # Vector of w_mean for *each* edge e:
    w_means = J_rec / jnp.sqrt(k_safe[indices])

    # Scales for normal distribution
    w_scales = 0.05 * w_means

    # 4) Sample random normal for each edge
    n_edges = indices.shape[0]
    randvals = jax.random.normal(key, shape=(n_edges,))

    # The new weight for edge e
    new_ws = w_means + randvals * w_scales

    return new_ws


class CSRConn(TwoEndConnector):
    """Connector built from the CSR sparse connection matrix. Same as brainpy, but without
    bounds checking to make it jax compatible"""

    def __init__(self, indices, inptr, **kwargs):
        super(CSRConn, self).__init__(**kwargs)

        self.indices = jnp.asarray(indices, dtype=get_idx_type())
        self.inptr = jnp.asarray(inptr, dtype=get_idx_type())
        self.pre_num = self.inptr.size - 1
        self.max_post = self.indices.max()

    def build_csr(self):
        # if self.pre_num != self.pre_num:
        #     raise ConnectorError(
        #         f"(pre_size, post_size) is inconsistent with "
        #         f"the shape of the sparse matrix."
        #     )
        # if self.post_num <= self.max_post:
        #     raise ConnectorError(
        #         f"post_num ({self.post_num}) should be greater than "
        #         f"the maximum id ({self.max_post}) of self.post_ids."
        #     )
        return self.indices, self.inptr

