import brainpy as bp
import brainpy.math as bm
from brainpy.connect import TwoEndConnector
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

from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from .utils import *


def euclidean_distance(pos1, pos2):
    return jnp.sqrt(jnp.sum((pos1 - pos2) ** 2))


class DistanceDependent(TwoEndConnector):
    """
    Synaptic connector that connects neurons based on their distances. Also, it caches the
    built matrix, so that multiple calls to requires('xxx') do not rebuild the connectivity
    matrix (for a given method 'xxx' like 'csr')

    Parameters
    ----------
    positions_pre : array-like
        Positions of the pre-synaptic neurons (list of tuples), shape (N_pre, D).
    positions_post : array-like, optional
        Positions of the post-synaptic neurons (list of tuples), shape (N_post, D).
        If not provided, defaults to positions_pre.
    domain : array-like, optional
        Size of the domain in each dimension, used for boundary conditions.
    boundary : str, optional
        Type of boundary conditions ('periodic', 'reflecting', 'absorbing').
    distance_metric : callable
        The distance metric to use, a function of two positions.
    kernel : callable
        The function defining the probability of connection based on distance.
    include_self : bool, optional
        Whether to include self-connections.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        kernel,
        domain,
        positions_pre,
        positions_post=None,
        boundary="periodic",
        distance_metric=euclidean_distance,
        include_self=False,
        seed=None,
        **kwargs,
    ):
        super(DistanceDependent, self).__init__(**kwargs)
        self.positions_pre = jnp.asarray(positions_pre)
        if positions_post is None:
            self.positions_post = self.positions_pre
        else:
            self.positions_post = jnp.asarray(positions_post)
        self.num_pre_neurons = self.positions_pre.shape[0]
        self.num_post_neurons = self.positions_post.shape[0]
        self.dimensions = self.positions_pre.shape[1]
        self.domain = jnp.asarray(domain)

        boundary_codes = {"periodic": 0, "reflecting": 1, "absorbing": 2}
        if boundary not in boundary_codes:
            raise ValueError(f"Unsupported boundary condition: {boundary}")
        self.boundary_code = boundary_codes[boundary]
        self.boundary = boundary  # For representation

        if distance_metric is None:
            raise ValueError("A distance metric function must be specified.")
        self.distance_metric = distance_metric

        if kernel is None:
            raise ValueError("A kernel function must be specified.")
        self.kernel = kernel

        self.include_self = include_self
        if seed is not None:
            self.key = seed
        else:
            self.key = jax.random.PRNGKey(np.random.randint(0, 2**32))

    def build_csr(self):
        # -------------------------------------------------------------------------
        # Helper: Compute distance with different boundary adjustments.
        @partial(jit, static_argnums=(2, 3))
        def compute_distance_with_boundary(
            pos_pre, pos_post, boundary_code, distance_metric
        ):
            # Compute differences
            delta = pos_pre - pos_post

            # Periodic boundary adjustment
            delta_periodic = (
                jnp.mod(delta + self.domain / 2, self.domain) - self.domain / 2
            )
            adjusted_pos_pre_periodic = pos_post + delta_periodic
            dist_periodic = distance_metric(adjusted_pos_pre_periodic, pos_post)

            # Reflecting boundary adjustment
            adjusted_pos_pre_reflecting = jnp.clip(pos_pre, 0, self.domain)
            adjusted_pos_post_reflecting = jnp.clip(pos_post, 0, self.domain)
            dist_reflecting = distance_metric(
                adjusted_pos_pre_reflecting, adjusted_pos_post_reflecting
            )

            # Absorbing boundary adjustment
            outside_pre = (pos_pre < 0) | (pos_pre > self.domain)
            outside_post = (pos_post < 0) | (pos_post > self.domain)
            outside = jnp.any(outside_pre) | jnp.any(outside_post)
            dist_absorbing = jnp.where(
                outside, jnp.inf, distance_metric(pos_pre, pos_post)
            )

            # No boundary adjustment
            dist_none = distance_metric(pos_pre, pos_post)

            distances = jnp.stack(
                [dist_periodic, dist_reflecting, dist_absorbing, dist_none]
            )
            return distances[boundary_code]

        # -------------------------------------------------------------------------
        # Helper: Compute distances from one pre–neuron to all post–neurons.
        def compute_row_distances(pos_pre):
            return vmap(
                lambda pos_post: compute_distance_with_boundary(
                    pos_pre, pos_post, self.boundary_code, self.distance_metric
                )
            )(self.positions_post)

        # -------------------------------------------------------------------------
        # To help JIT compilation, capture some attributes as local variables.
        include_self = self.include_self
        num_pre_neurons = self.num_pre_neurons
        num_post_neurons = self.num_post_neurons
        domain = (
            self.domain
        )  # if needed inside compute_distance (already used via self)

        # -------------------------------------------------------------------------
        # Jitted inner function: process a chunk of pre–neurons.
        # (Note: We removed the jnp.where for the connections indices from inside the JIT.)
        @partial(jit, static_argnums=(2,))
        def process_chunk_inner(pre_chunk, key, start):
            # Compute distances for all rows in this chunk (via double vmap)
            distances = vmap(compute_row_distances)(
                pre_chunk
            )  # shape: (chunk_size, num_post_neurons)
            probs = self.kernel(distances)
            key, subkey = jax.random.split(key)
            rand_vals = jax.random.uniform(subkey, shape=probs.shape)
            connections = probs > rand_vals

            # Remove self–connections if needed.
            if not include_self:
                # For each row in the chunk, the global index is start + row_index.
                chunk_len = pre_chunk.shape[0]
                global_indices = jnp.arange(start, start + chunk_len)
                min_neurons = jnp.minimum(num_pre_neurons, num_post_neurons)
                rows = jnp.arange(chunk_len)
                new_vals = jnp.where(
                    global_indices < min_neurons,
                    False,
                    connections[rows, global_indices],
                )
                connections = connections.at[rows, global_indices].set(new_vals)

            counts = jnp.sum(
                connections, axis=1
            )  # number of connections per row in this chunk
            return key, connections, counts

        # -------------------------------------------------------------------------
        # Loop over pre–neurons in chunks.
        CHUNK_SIZE = 2**14  # Adjust as appropriate.
        num_pre = self.num_pre_neurons
        key = self.key  # current random key

        pre_idx_list = []  # will hold global row indices of connections
        post_idx_list = []  # will hold column indices of connections
        count_list = []  # per–row connection counts

        # Process each chunk sequentially.
        for start in range(0, num_pre, CHUNK_SIZE):
            pre_chunk = self.positions_pre[start : start + CHUNK_SIZE]
            key, connections, counts = process_chunk_inner(pre_chunk, key, start)
            # --- IMPORTANT: Do the nonzero (jnp.where) outside the JIT to avoid dynamic shape issues.
            chunk_pre_idx, chunk_post_idx = jnp.where(connections)
            global_pre_idx = (
                chunk_pre_idx + start
            )  # convert local (chunk) row indices to global ones
            pre_idx_list.append(global_pre_idx)
            post_idx_list.append(chunk_post_idx)
            count_list.append(counts)

        # Update the random key.
        self.key = key

        # Concatenate results from all chunks.
        pre_ids = jnp.concatenate(pre_idx_list)
        post_ids = jnp.concatenate(post_idx_list)
        row_counts = jnp.concatenate(count_list)
        indptr = jnp.concatenate(
            [jnp.array([0], dtype=row_counts.dtype), jnp.cumsum(row_counts)]
        )

        # Cast the indices to the desired index type and return.
        return post_ids.astype(get_idx_type()), indptr.astype(get_idx_type())

    def to_dict(self, keys=["boundary", "include_self"]):
        out = {key: value for key, value in self.__dict__.items() if key in keys}
        out["kernel"] = {self.kernel.__class__.__name__: self.kernel.to_dict()}
        return out


class AbstractKernel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass


class GaussianKernel(AbstractKernel):
    """
    Gaussian kernel for a distance-dependent connector.

    Parameters
    ----------
    sigma : float, keyword-only
        Width of the Gaussian.
    p_max : float, optional
        Maximum probability of connection.
    """

    def __init__(
        self,
        sigma,
        p_max=1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.p_max = p_max

    def mass2pmax(omega, sigma):
        """
        Convert the total integrated distribution to the height at the origin. Only for 2D.
        """
        return omega / (2 * np.pi * sigma**2)

    def pmax2mass(p_max, sigma):
        """
        Convert the height at the origin to the total integrated distribution. Only for 2D.
        """
        return p_max * (2 * np.pi * sigma**2)

    def __call__(self, distance):
        return self.p_max * jnp.exp(-(distance**2) / (2 * self.sigma**2))

    def to_dict(self):
        return {"sigma": self.sigma, "p_max": self.p_max}


class ExponentialKernel(AbstractKernel):
    """
    Exponential kernel for a distance-dependent connector.

    Parameters
    ----------
    sigma : float, keyword-only
        Decay constant (length scale) for the exponential kernel.
    p_max : float, optional
        Maximum probability of connection (default is 1.0).
    """

    def __init__(self, sigma, p_max=1.0):
        super().__init__()
        self.sigma = sigma
        self.p_max = p_max

    def __call__(self, distance):
        """
        Compute the connection probability given a distance.

        Parameters
        ----------
        distance : float or array-like
            The distance(s) between pre- and post-synaptic neurons.

        Returns
        -------
        float or array-like
            The connection probability.
        """
        return self.p_max * jnp.exp(-distance / self.sigma)

    def mass2pmax(omega, sigma):
        """
        Convert the total integrated distribution to the height at the origin. Only for 2D.
        """
        return omega / (2 * np.pi * sigma**2)

    def pmax2mass(p_max, sigma):
        """
        Convert the height at the origin to the total integrated distribution. Only for 2D.
        """
        return p_max * (2 * np.pi * sigma**2)

    def to_dict(self):
        """
        Return a dictionary representation of the kernel's parameters.
        """
        return {"sigma": self.sigma, "p_max": self.p_max}


class FixedProb(TwoEndConnector):
    def __init__(
        self,
        prob,
        pre_ratio=1.0,
        include_self=True,
        seed=None,
        **kwargs,
    ):
        super(FixedProb, self).__init__(**kwargs)
        assert 0.0 <= prob <= 1.0
        assert 0.0 <= pre_ratio <= 1.0
        self.prob = prob
        self.pre_ratio = pre_ratio
        self.include_self = include_self
        self.seed = seed
        self._jaxrand = bm.random.default_rng(self.seed)

    def _iii(self):
        if (not self.include_self) and (self.pre_num != self.post_num):
            raise bp.ConnectorError(
                f"We found pre_num != post_num ({self.pre_num} != {self.post_num}). "
                f"But `include_self` is set to True."
            )

        if self.pre_ratio < 1.0:
            pre_num_to_select = int(self.pre_num * self.pre_ratio)
            pre_ids = self._jaxrand.choice(
                self.pre_num, size=(pre_num_to_select,), replace=False
            )
        else:
            pre_num_to_select = self.pre_num
            pre_ids = jnp.arange(self.pre_num)

        post_num_total = self.post_num
        post_num_to_select = int(self.post_num * self.prob)

        selected_post_ids = self._jaxrand.randint(
            0, post_num_total, (pre_num_to_select, post_num_to_select)
        )

        return (
            pre_num_to_select,
            post_num_to_select,
            bm.as_jax(selected_post_ids),
            bm.as_jax(pre_ids),
        )

    def build_csr(self):
        pre_num_to_select, post_num_to_select, selected_post_ids, pre_ids = self._iii()
        pre_nums = jnp.ones(pre_num_to_select) * post_num_to_select
        if not self.include_self:
            true_ids = selected_post_ids == jnp.reshape(pre_ids, (-1, 1))
            pre_nums -= jnp.sum(true_ids, axis=1)
            selected_post_ids = selected_post_ids.flatten()[
                jnp.logical_not(true_ids).flatten()
            ]
        else:
            selected_post_ids = selected_post_ids.flatten()
        selected_pre_inptr = jnp.cumsum(jnp.concatenate([jnp.zeros(1), pre_nums]))
        return selected_post_ids.astype(get_idx_type()), selected_pre_inptr.astype(
            get_idx_type()
        )
