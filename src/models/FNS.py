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
from scipy.integrate import dblquad

import brainpy as bp
import brainpy.math as bm

from brainpy.dyn.neurons import GradNeuDyn
from brainpy.dyn import NeuDyn
from brainpy.initialize import (
    ZeroInit,
    OneInit,
    Uniform,
    variable_,
    noise as init_noise,
)
from brainpy import share
from brainpy.types import Shape, ArrayType
from brainpy.check import is_initializer
from brainpy import odeint, sdeint, JointEq
from brainpy._src.connect.base import get_idx_type
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from ..neurons import FNSNeuron
from ..positions import *
from ..synapses import *


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

    # def build_mat(self):
    #     # Capture constants to pass into JIT-compiled functions
    #     boundary_code = self.boundary_code
    #     domain = self.domain
    #     distance_metric = self.distance_metric

    #     @partial(jit, static_argnums=(2, 3))
    #     def compute_distance_with_boundary(
    #         pos_pre, pos_post, boundary_code, distance_metric
    #     ):
    #         # Adjust positions according to boundary conditions

    #         delta = pos_pre - pos_post

    #         # Periodic boundary adjustment
    #         delta_periodic = jnp.mod(delta + domain / 2, domain) - domain / 2
    #         adjusted_pos_pre_periodic = pos_post + delta_periodic
    #         adjusted_pos_post_periodic = pos_post
    #         dist_periodic = distance_metric(
    #             adjusted_pos_pre_periodic, adjusted_pos_post_periodic
    #         )

    #         # Reflecting boundary adjustment
    #         adjusted_pos_pre_reflecting = jnp.clip(pos_pre, 0, domain)
    #         adjusted_pos_post_reflecting = jnp.clip(pos_post, 0, domain)
    #         dist_reflecting = distance_metric(
    #             adjusted_pos_pre_reflecting, adjusted_pos_post_reflecting
    #         )

    #         # Absorbing boundary adjustment
    #         outside_pre = (pos_pre < 0) | (pos_pre > domain)
    #         outside_post = (pos_post < 0) | (pos_post > domain)
    #         outside = jnp.any(outside_pre) | jnp.any(outside_post)
    #         dist_absorbing = distance_metric(pos_pre, pos_post)
    #         dist_absorbing = jnp.where(outside, jnp.inf, dist_absorbing)

    #         # None boundary adjustment
    #         dist_none = distance_metric(pos_pre, pos_post)

    #         # Stack distances
    #         distances = jnp.stack(
    #             [dist_periodic, dist_reflecting, dist_absorbing, dist_none]
    #         )

    #         # Select the appropriate distance based on boundary_code
    #         dist = distances[boundary_code]
    #         return dist

    #     @partial(jit, static_argnums=(2, 3))
    #     def compute_probability(pos_pre, pos_post, kernel, distance_metric):
    #         dist = compute_distance_with_boundary(
    #             pos_pre, pos_post, boundary_code, distance_metric
    #         )
    #         prob = kernel(dist)
    #         return prob

    #     # Fix kernel and distance_metric using partial to ensure they are static
    #     compute_probability_fixed = partial(
    #         compute_probability, kernel=self.kernel, distance_metric=distance_metric
    #     )

    #     # Vectorize the computation
    #     compute_probability_vmap = vmap(
    #         vmap(
    #             compute_probability_fixed,
    #             in_axes=(None, 0),
    #         ),
    #         in_axes=(0, None),
    #     )

    #     probabilities = compute_probability_vmap(
    #         self.positions_pre,
    #         self.positions_post,
    #     )

    #     probabilities = jnp.clip(probabilities, 0.0, 1.0)

    #     # Generate random numbers
    #     self.key, subkey = jax.random.split(self.key)
    #     random_matrix = jax.random.uniform(subkey, shape=probabilities.shape)

    #     conn_mat = probabilities > random_matrix

    #     # Remove self-connections if include_self is False
    #     if not self.include_self:
    #         min_neurons = min(self.num_pre_neurons, self.num_post_neurons)
    #         diag_indices = (jnp.arange(min_neurons), jnp.arange(min_neurons))
    #         conn_mat = conn_mat.at[diag_indices].set(False)
    #     return conn_mat
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
        self.sigma = sigma
        self.p_max = p_max

        # super(GaussianKernel, self).__init__(
        #     domain=domain,
        #     positions_pre=pre_positions,
        #     positions_post=post_positions,
        #     kernel=gaussian_kernel,
        #     **kwargs,
        # )

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
        self.sigma = sigma
        self.p_max = p_max

        # You might want to call the parent class's __init__ if necessary.
        # For example:
        # super(ExponentialKernel, self).__init__(
        #     domain=domain,
        #     positions_pre=pre_positions,
        #     positions_post=post_positions,
        #     kernel=exponential_kernel,
        #     **kwargs,
        # )

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

    def to_dict(self):
        """
        Return a dictionary representation of the kernel's parameters.
        """
        return {"sigma": self.sigma, "p_max": self.p_max}


def expected_indegree(FNSnet, pop="ee", approx=True):
    """
    Compute the mean indegree.
    Should converge to `2 * np.pi * sigma**2 * p_max * rho` in the limit of large dx/small sigma.
    """
    assert FNSnet.boundary == "periodic"
    dx = FNSnet.dx
    if pop[0] == "e":
        rho = FNSnet.rho
    else:
        rho = FNSnet.rho / FNSnet.gamma
    if pop == "ee":
        p_max = FNSnet.p_ee
        sigma = FNSnet.sigma_ee
    elif pop == "ei":
        p_max = FNSnet.p_ei
        sigma = FNSnet.sigma_ei
    elif pop == "ie":
        p_max = FNSnet.p_ie
        sigma = FNSnet.sigma_ie
    elif pop == "ii":
        p_max = FNSnet.p_ii
        sigma = FNSnet.sigma_ii

    if approx:
        return 2 * np.pi * sigma**2 * p_max * rho
    else:

        def integrand(y, x, dx, sigma, p_max):
            # y is integrated first, x second (dblquad's calling convention).
            rx = min(x, dx - x)
            ry = min(y, dx - y)
            r2 = rx * rx + ry * ry
            return p_max * np.exp(-r2 / (2.0 * sigma * sigma))

        result, error_est = dblquad(
            integrand,
            0,
            dx,  # outer integral range for x
            lambda x: 0,  # lower limit for y
            lambda x: dx,  # upper limit for y
            args=(dx, sigma, p_max),
        )
        return rho * result


def indegrees(proj):  # ! NOT JAX COMPATIBLE???
    N = jnp.prod(jnp.array(proj.post.size))
    indices = jnp.array(proj.comm.indices, dtype=int)
    in_degree = jnp.bincount(indices, length=N)
    return in_degree


def indegree(proj):
    N = jnp.prod(jnp.array(proj.post.size))
    indices = jnp.array(proj.comm.indices, dtype=int)
    in_degree = jnp.bincount(indices, length=N)
    return jnp.mean(in_degree)


def correlate_weights(proj, J, key):
    k = indegrees(proj)

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


class FNScircuit(bp.Network):
    def __init__(
        self,
        rho=30000,  # Density of Exc. neurons (neurons per mm^2)
        dx=1.0,  # Width of the spatial domain (mm)
        sigma_ee=0.125,  # Width of the distance-dependent connectivity kernel (mm)
        sigma_ei=0.1,
        sigma_ie=0.1,
        sigma_ii=0.125,
        p_ee=0.1,  # Maximum connection probability (Campagnola2022, corrected)
        p_ei=0.2,
        p_ie=0.3,
        p_ii=0.3,
        boundary="periodic",
        include_self=False,
        gamma=4,  # Ratio of num. Exc. to num. Inh. neurons
        zeta=4,  # Per-neuron synaptic weight I:E ratio
        nu=1,  # External population firing rate
        n_ext=10,  # Number of external synapses per Exc. neuron
        J_e=0.0004,  # ! Currently abitrary. Has same units as g_L? uS
        kernel=GaussianKernel,
        method="exp_auto",
        key=jax.random.PRNGKey(np.random.randint(0, 2**32)),
        copy_conn=False,  # Whether to copy connectivity from the provided FNScircuit
    ):
        super().__init__()

        self.rho = rho
        self.dx = dx
        self.sigma_ee = sigma_ee
        self.sigma_ei = sigma_ei
        self.sigma_ie = sigma_ie
        self.sigma_ii = sigma_ii
        self.p_ee = p_ee
        self.p_ei = p_ei
        self.p_ie = p_ie
        self.p_ii = p_ii
        self.boundary = boundary
        self.include_self = include_self
        self.gamma = gamma
        self.zeta = zeta
        self.nu = nu
        self.n_ext = n_ext
        self.J_e = J_e
        self.J_i = (
            self.J_e * zeta
        )  # !!! Not negative, because the reversal threshold for the inhibitory synapses is negative
        self.method = method
        self.kernel = kernel
        self.key = key

        # geometry
        A = dx**2
        ne = round(np.sqrt(rho * A))  # Number of grid points in each dimension
        ni = round((ne**2) / gamma)

        # dx bounds the grid, assuming left bottom corner is at (0, 0)
        exc_positions = GridPositions((dx, dx))

        self.key, subkey = jax.random.split(self.key)
        inh_positions = RandomPositions((dx, dx), subkey)

        # neurons
        self.key, subkey = jax.random.split(self.key)
        self.E = FNSNeuron(
            size=(ne, ne),
            C=0.25,
            g_L=0.0167,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=60.0,
            Delta_g_K=0.002,
            V_initializer=bp.init.Uniform(-85.0, -50.0, subkey),
            method=method,
            embedding=exc_positions,
        )

        # Create a population of inhibitory neurons
        self.key, subkey = jax.random.split(self.key)
        self.I = FNSNeuron(
            size=ni,
            C=0.25,
            g_L=0.025,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-70.0,
            tau_ref=4.0,
            V_K=-85.0,
            tau_K=60.0,
            Delta_g_K=0.0,  # No adaptation for inhibitory neurons
            V_initializer=bp.init.Uniform(-55.0, -50.0, subkey),
            method=method,
            embedding=inh_positions,
        )

        # Connectivity topology

        # External population
        p_ext = np.sqrt(self.n_ext / self.E.num)  # !!! Check !!!
        N_ext = int(np.round(self.E.num * p_ext))

        def copy_connectivity(proj):
            # if isinstance(proj, bp.Projection):
            #     indices = copy.deepcopy(proj.comm.indices)
            #     inptr = copy.deepcopy(proj.comm.indptr)
            # else:
            indices = copy.deepcopy(proj["indices"])
            inptr = copy.deepcopy(proj["indptr"])
            return CSRConn(indices, inptr)

        if copy_conn:
            if isinstance(copy_conn, FNScircuit):
                copy_conn = copy_conn.get_connectivity()
            conn_ee = copy_connectivity(copy_conn["E2E"])
            conn_ei = copy_connectivity(copy_conn["E2I"])
            conn_ie = copy_connectivity(copy_conn["I2E"])
            conn_ii = copy_connectivity(copy_conn["I2I"])
            conn_exte = copy_connectivity(copy_conn["ext2E"])
            conn_exti = copy_connectivity(copy_conn["ext2I"])

        else:
            self.key, subkey = jax.random.split(self.key)
            conn_ee = DistanceDependent(
                kernel=kernel(sigma=sigma_ee, p_max=p_ee),
                domain=self.E.embedding.domain,
                positions_pre=self.E.positions,
                positions_post=self.E.positions,
                boundary=boundary,
                include_self=include_self,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ei = DistanceDependent(
                kernel=kernel(sigma=sigma_ei, p_max=p_ei),
                domain=self.E.embedding.domain,
                positions_pre=self.E.positions,
                positions_post=self.I.positions,
                boundary=boundary,
                include_self=include_self,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ie = DistanceDependent(
                kernel=kernel(sigma=sigma_ie, p_max=p_ie),
                domain=self.I.embedding.domain,
                positions_pre=self.I.positions,
                positions_post=self.E.positions,
                boundary=boundary,
                include_self=include_self,
                seed=subkey,
            )
            self.key, subkey = jax.random.split(self.key)
            conn_ii = DistanceDependent(
                kernel=kernel(sigma=sigma_ii, p_max=p_ii),
                domain=self.I.embedding.domain,
                positions_pre=self.I.positions,
                positions_post=self.I.positions,
                boundary=boundary,
                include_self=include_self,
                seed=subkey,
            )

            self.key, subkey = jax.random.split(self.key)
            conn_exte = bp.connect.FixedProb(
                prob=p_ext, allow_multi_conn=True, seed=subkey
            )
            self.key, subkey = jax.random.split(self.key)
            conn_exti = bp.connect.FixedProb(
                prob=p_ext, allow_multi_conn=True, seed=subkey
            )

        # Synapses
        tau_d_e = 5.0  # * Excitatory synapse decays more slowly than inhibitory
        tau_d_i = 4.5
        V_rev_e = 0.0
        V_rev_i = -80.0  # ? Makes the inhibitory synapses inhibitory

        self.E2E = Synapse(
            pre=self.E,
            post=self.E,
            delay=2.0,
            conn=conn_ee,
            tau_d=tau_d_e,
            tau_r=1.0,
            g_max=bp.init.Normal(self.J_e, self.J_e * 0.05),
            V_rev=V_rev_e,
        )

        self.E2I = Synapse(
            pre=self.E,
            post=self.I,
            delay=2.0,
            conn=conn_ei,
            tau_d=tau_d_e,
            tau_r=1.0,
            g_max=bp.init.Normal(self.J_e, self.J_e * 0.05),
            V_rev=V_rev_e,
        )

        self.I2E = Synapse(
            pre=self.I,
            post=self.E,
            delay=2.0,
            conn=conn_ie,
            tau_d=tau_d_i,
            tau_r=1.0,
            g_max=bp.init.Normal(self.J_i, self.J_i * 0.05),
            V_rev=V_rev_i,
        )

        self.I2I = Synapse(
            pre=self.I,
            post=self.I,
            delay=2.0,
            conn=conn_ii,
            tau_d=tau_d_i,
            tau_r=1.0,
            g_max=bp.init.Normal(self.J_i, self.J_i * 0.05),
            V_rev=V_rev_i,
        )

        # External population
        self.key, subkey = jax.random.split(self.key)
        self.ext = bp.dyn.PoissonGroup(
            size=N_ext,
            freqs=self.nu,
            keep_size=False,
            sharding=None,
            spk_type=None,
            name=None,
            mode=None,
            seed=subkey,
        )
        self.ext2E = Synapse(
            pre=self.ext,
            post=self.E,
            conn=conn_exte,
            delay=2.0,
            tau_d=tau_d_e,
            g_max=self.J_e,
        )
        self.ext2I = Synapse(
            pre=self.ext,
            post=self.I,
            conn=conn_exti,
            delay=2.0,
            tau_d=tau_d_e,
            g_max=self.J_e,
        )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

        # * Posthoc weight updates to maintain mean_weight = 1/sqrt(in-degree) per neuron
        self.reinit_weights(self.zeta, self.J_e)  # !! Need to fix it seems

    def reinit_weights(self, zeta=None, J_e=None):
        if zeta:
            self.zeta = zeta
        if J_e:
            self.J_e = J_e
        self.J_i = self.J_e * self.zeta
        self.key, subkey = jax.random.split(self.key)
        self.E2E.proj.comm.weight = correlate_weights(self.E2E.proj, self.J_e, subkey)
        self.key, subkey = jax.random.split(self.key)
        self.E2I.proj.comm.weight = correlate_weights(self.E2I.proj, self.J_e, subkey)
        self.key, subkey = jax.random.split(self.key)
        self.I2E.proj.comm.weight = correlate_weights(self.I2E.proj, self.J_i, subkey)
        self.key, subkey = jax.random.split(self.key)
        self.I2I.proj.comm.weight = correlate_weights(self.I2I.proj, self.J_i, subkey)
        self.key, subkey = jax.random.split(self.key)
        self.ext2E.proj.comm.weight = correlate_weights(
            self.ext2E.proj, self.J_e, subkey
        )
        self.key, subkey = jax.random.split(self.key)
        self.ext2I.proj.comm.weight = correlate_weights(
            self.ext2I.proj, self.J_e, subkey
        )
        self.reset_state()  ## or bp.reset_state(self)??

    def reinit_nu(self, nu):
        self.nu = nu
        self.ext.freqs = nu
        self.reset_state()

    def get_connectivity(self):
        # Extract just the connectivity indices and inptrs
        return {
            "E2E": {
                "indices": self.E2E.proj.comm.indices,
                "indptr": self.E2E.proj.comm.indptr,
            },
            "E2I": {
                "indices": self.E2I.proj.comm.indices,
                "indptr": self.E2I.proj.comm.indptr,
            },
            "I2E": {
                "indices": self.I2E.proj.comm.indices,
                "indptr": self.I2E.proj.comm.indptr,
            },
            "I2I": {
                "indices": self.I2I.proj.comm.indices,
                "indptr": self.I2I.proj.comm.indptr,
            },
            "ext2E": {
                "indices": self.ext2E.proj.comm.indices,
                "indptr": self.ext2E.proj.comm.indptr,
            },
            "ext2I": {
                "indices": self.ext2I.proj.comm.indices,
                "indptr": self.ext2I.proj.comm.indptr,
            },
        }

    def get_input_params(self):
        keys = [
            "rho",
            "dx",
            "sigma_ee",
            "sigma_ei",
            "sigma_ie",
            "sigma_ii",
            "p_ee",
            "p_ei",
            "p_ie",
            "p_ii",
            "boundary",
            "include_self",
            "gamma",
            "zeta",
            "nu",
            "n_ext",
            "J_e",
            "kernel",
            "method",
            "key",
        ]
        _params = {key: value for key, value in self.__dict__.items() if key in keys}
        if len(_params) != len(keys):
            missings = set(keys) - set(_params.keys())
            raise ValueError("Missing parameters: {}".format(missings))
        return _params

    def update_copy(self, **params):
        _params = self.get_input_params()
        params = {**_params, **params}
        new_model = self.__class__(copy_conn=self, **params)
        new_model.reinit_weights(params["zeta"], params["J_e"])
        new_model.reinit_nu(params["nu"])
        bp.reset_state(new_model)
        return new_model

    def to_dict(
        self,
        keys=[
            "rho",
            "dx",
            "sigma_ee",
            "sigma_ei",
            "sigma_ie",
            "sigma_ii",
            "p_ee",
            "p_ei",
            "p_ie",
            "p_ii",
            "boundary",
            "include_self",
            "gamma",
            "g",
            "nu",
            "p_ext",
            "J_e",
            "method",
        ],
    ):
        out = {
            "parameters": {
                key: maybe_initializer(value)
                for key, value in self.__dict__.items()
                if key in keys
            }
        }
        out["populations"] = {
            "E": self.E.to_dict(),
            "I": self.I.to_dict(),
        }
        out["synapses"] = {
            "E2E": self.E2E.to_dict(),
            "E2I": self.E2I.to_dict(),
            "I2E": self.I2E.to_dict(),
            "I2I": self.I2I.to_dict(),
        }
        return {self.__class__.__name__: out}
