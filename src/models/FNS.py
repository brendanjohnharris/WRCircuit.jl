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
import uuid

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
        include_self=True,
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
        if seed is not None:  # Want to pass in the model uuid for reproducibility
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 2**32)
        self.key = jax.random.PRNGKey(self.seed)

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
        # Precompute distances
        @partial(jit, static_argnums=(2, 3))
        def compute_distance_with_boundary(
            pos_pre, pos_post, boundary_code, distance_metric
        ):
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

            # None boundary adjustment
            dist_none = distance_metric(pos_pre, pos_post)

            # Select the appropriate distance based on boundary_code
            distances = jnp.stack(
                [dist_periodic, dist_reflecting, dist_absorbing, dist_none]
            )
            return distances[boundary_code]

        # Vectorize distance computation
        compute_distance_vmap = vmap(
            vmap(compute_distance_with_boundary, in_axes=(None, 0, None, None)),
            in_axes=(0, None, None, None),
        )

        # Precompute the distance matrix
        distance_matrix = compute_distance_vmap(
            self.positions_pre,
            self.positions_post,
            self.boundary_code,
            self.distance_metric,
        )

        # Compute probabilities using the kernel
        probabilities = self.kernel(distance_matrix)

        # Generate random numbers only for non-zero probabilities
        self.key, subkey = jax.random.split(self.key)
        random_values = jax.random.uniform(subkey, shape=probabilities.shape)
        connections = probabilities > random_values

        # Remove self-connections if needed
        if not self.include_self:
            min_neurons = min(self.num_pre_neurons, self.num_post_neurons)
            diag_indices = (jnp.arange(min_neurons), jnp.arange(min_neurons))
            connections = connections.at[diag_indices].set(False)

        # Convert to sparse CSR format
        pre_ids, post_ids = jnp.where(connections)

        # Ensure indices are within bounds
        assert jnp.all(pre_ids < self.num_pre_neurons), "pre_ids out of bounds"
        assert jnp.all(post_ids < self.num_post_neurons), "post_ids out of bounds"

        # Compute the number of non-zero elements per row
        pre_nums = jnp.bincount(pre_ids, length=self.num_pre_neurons)

        # Compute the indptr array
        indptrs = jnp.concatenate([jnp.array([0]), jnp.cumsum(pre_nums)])

        # # Debug: Print intermediate results
        # print("post_ids (column indices):", post_ids)
        # print("pre_ids (row indices):", pre_ids)
        # print("pre_nums (non-zero counts per row):", pre_nums)
        # print("indptrs (CSR indptr array):", indptrs)

        return post_ids.astype(get_idx_type()), indptrs.astype(get_idx_type())

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


class FNScircuit(bp.Network):
    def __init__(
        self,
        rho=30000,  # Density of Exc. neurons (neurons per mm^2)
        dx=1,  # Width of the spatial domain (mm)
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
        gamma=4,  # Ratio of Exc. to Inh. neurons
        g=4,  # Per-neuron synaptic weight E:I ratio
        nu=1,  # External population firing rate
        p_ext=0.1,
        J_e=0.2,  # ! Currently abritrary
        method="exp_auto",
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
        self.g = g
        self.nu = nu
        self.p_ext = p_ext
        self.J_e = J_e
        self.J_i = (
            g * self.J_e
        )  # !!! Not negative, because the reversal threshold for the inhibitory synapses is negative
        self.method = method

        # geometry
        A = dx**2
        ne = round(np.sqrt(rho * A))  # Number of grid points in each dimension
        ni = round((ne**2) / gamma)

        # dx bounds the grid, assuming left bottom corner is at (0, 0)
        exc_positions = GridPositions((dx, dx))
        inh_positions = RandomPositions((dx, dx))

        # neurons
        self.E = FNSNeuron(
            size=(ne, ne),
            C=0.25,
            g_L=0.0167,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-60.0,
            tau_ref=4.0,
            tau_K=80.0,
            V_initializer=bp.init.Uniform(-70.0, -50.0),
            method=method,
            embedding=exc_positions,
        )

        # Create a population of inhibitory neurons
        self.I = FNSNeuron(
            size=ni,
            C=0.25,
            g_L=0.0167,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-60.0,
            tau_ref=4.0,
            tau_K=80.0,
            V_initializer=bp.init.Uniform(-70.0, -50.0),
            method=method,
            embedding=inh_positions,
        )

        # Connectivity topology
        conn_ee = DistanceDependent(
            kernel=GaussianKernel(sigma=sigma_ee, p_max=p_ee),
            domain=self.E.embedding.domain,
            positions_pre=self.E.positions,
            positions_post=self.E.positions,
            boundary=boundary,
            include_self=include_self,
        )
        conn_ei = DistanceDependent(
            kernel=GaussianKernel(sigma=sigma_ei, p_max=p_ei),
            domain=self.E.embedding.domain,
            positions_pre=self.E.positions,
            positions_post=self.I.positions,
            boundary=boundary,
            include_self=include_self,
        )
        conn_ie = DistanceDependent(
            kernel=GaussianKernel(sigma=sigma_ie, p_max=p_ie),
            domain=self.I.embedding.domain,
            positions_pre=self.I.positions,
            positions_post=self.E.positions,
            boundary=boundary,
            include_self=include_self,
        )
        conn_ii = DistanceDependent(
            kernel=GaussianKernel(sigma=sigma_ii, p_max=p_ii),
            domain=self.I.embedding.domain,
            positions_pre=self.I.positions,
            positions_post=self.I.positions,
            boundary=boundary,
            include_self=include_self,
        )

        # Synapses
        tau_d_e = 4.0  # ! Maybe these are different? E2I gets tau_d_e?
        tau_d_i = 1.0
        V_rev_e = 0.0
        V_rev_i = -80  # ? Makes the inhibitory synapses inhibitory
        self.E2E = Synapse(
            pre=self.E,
            post=self.E,
            delay=2.0,
            conn=conn_ee,
            tau_d=tau_d_e,
            tau_r=1.0,
            g_max=self.J_e,
            V_rev=V_rev_e,
        )
        self.E2I = Synapse(
            pre=self.E,
            post=self.I,
            delay=2.0,
            conn=conn_ei,
            tau_d=tau_d_e,  # ! Is this tau_d_e or tau_d_i?
            tau_r=1.0,
            g_max=self.J_e,
            V_rev=V_rev_e,
        )
        self.I2E = Synapse(
            pre=self.I,
            post=self.E,
            delay=2.0,
            conn=conn_ie,
            tau_d=tau_d_i,
            tau_r=1.0,
            g_max=self.J_i,
            V_rev=V_rev_i,
        )
        self.I2I = Synapse(
            pre=self.I,
            post=self.I,
            delay=2.0,
            conn=conn_ii,
            tau_d=tau_d_i,
            tau_r=1.0,
            g_max=self.J_i,
            V_rev=V_rev_i,
        )

        # External population
        # self.ext = bp.dyn.PoissonGroup(
        #     size=int(
        #         np.round(self.E.num * p_ext)
        #     ),  # So that the average number of connections to each population matches Ce
        #     freqs=self.nu,
        #     keep_size=False,
        #     sharding=None,
        #     spk_type=None,
        #     name=None,
        #     mode=None,
        #     seed=None,
        # )
        # self.ext2E = Synapse(
        #     pre=self.ext,
        #     post=self.E,
        #     conn=bp.connect.FixedProb(prob=p_ext, allow_multi_conn=True),
        #     delay=2.0,
        #     tau_d=5.0,
        #     g_max=self.J_e,
        # )
        # self.ext2I = Synapse(
        #     pre=self.ext,
        #     post=self.I,
        #     conn=bp.connect.FixedProb(prob=p_ext, allow_multi_conn=True),
        #     delay=2.0,
        #     tau_d=3.0,
        #     g_max=self.J_e,
        # )

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

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
