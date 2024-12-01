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
from typing import Union, Callable, Optional, Sequence, Any
from functools import partial
from ..neurons import FNSNeuron
from ..positions import *


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
    Synaptic connector that connects neurons based on their distances.

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

    def build_mat(self):
        # Capture constants to pass into JIT-compiled functions
        boundary_code = self.boundary_code
        domain = self.domain
        distance_metric = self.distance_metric

        @partial(jit, static_argnums=(2, 3))
        def compute_distance_with_boundary(
            pos_pre, pos_post, boundary_code, distance_metric
        ):
            # Adjust positions according to boundary conditions

            delta = pos_pre - pos_post

            # Periodic boundary adjustment
            delta_periodic = jnp.mod(delta + domain / 2, domain) - domain / 2
            adjusted_pos_pre_periodic = pos_post + delta_periodic
            adjusted_pos_post_periodic = pos_post
            dist_periodic = distance_metric(
                adjusted_pos_pre_periodic, adjusted_pos_post_periodic
            )

            # Reflecting boundary adjustment
            adjusted_pos_pre_reflecting = jnp.clip(pos_pre, 0, domain)
            adjusted_pos_post_reflecting = jnp.clip(pos_post, 0, domain)
            dist_reflecting = distance_metric(
                adjusted_pos_pre_reflecting, adjusted_pos_post_reflecting
            )

            # Absorbing boundary adjustment
            outside_pre = (pos_pre < 0) | (pos_pre > domain)
            outside_post = (pos_post < 0) | (pos_post > domain)
            outside = jnp.any(outside_pre) | jnp.any(outside_post)
            dist_absorbing = distance_metric(pos_pre, pos_post)
            dist_absorbing = jnp.where(outside, jnp.inf, dist_absorbing)

            # None boundary adjustment
            dist_none = distance_metric(pos_pre, pos_post)

            # Stack distances
            distances = jnp.stack(
                [dist_periodic, dist_reflecting, dist_absorbing, dist_none]
            )

            # Select the appropriate distance based on boundary_code
            dist = distances[boundary_code]
            return dist

        @partial(jit, static_argnums=(2, 3))
        def compute_probability(pos_pre, pos_post, kernel, distance_metric):
            dist = compute_distance_with_boundary(
                pos_pre, pos_post, boundary_code, distance_metric
            )
            prob = kernel(dist)
            return prob

        # Fix kernel and distance_metric using partial to ensure they are static
        compute_probability_fixed = partial(
            compute_probability, kernel=self.kernel, distance_metric=distance_metric
        )

        # Vectorize the computation
        compute_probability_vmap = vmap(
            vmap(
                compute_probability_fixed,
                in_axes=(None, 0),
            ),
            in_axes=(0, None),
        )

        probabilities = compute_probability_vmap(
            self.positions_pre,
            self.positions_post,
        )

        probabilities = jnp.clip(probabilities, 0.0, 1.0)

        # Generate random numbers
        self.key, subkey = jax.random.split(self.key)
        random_matrix = jax.random.uniform(subkey, shape=probabilities.shape)

        conn_mat = probabilities > random_matrix

        # Remove self-connections if include_self is False
        if not self.include_self:
            min_neurons = min(self.num_pre_neurons, self.num_post_neurons)
            diag_indices = (jnp.arange(min_neurons), jnp.arange(min_neurons))
            conn_mat = conn_mat.at[diag_indices].set(False)

        return conn_mat

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
    max_prob : float, optional
        Maximum probability of connection.
    """

    def __init__(
        self,
        sigma,
        max_prob=1.0,
    ):
        self.sigma = sigma
        self.max_prob = max_prob

        # super(GaussianKernel, self).__init__(
        #     domain=domain,
        #     positions_pre=pre_positions,
        #     positions_post=post_positions,
        #     kernel=gaussian_kernel,
        #     **kwargs,
        # )

    def __call__(self, distance):
        return self.max_prob * jnp.exp(-(distance**2) / (2 * self.sigma**2))

    def to_dict(self):
        return {"sigma": self.sigma, "max_prob": self.max_prob}


class FNSCircuit(bp.Network):
    def __init__(self, num_exc, num_inh, method="exp_auto"):
        super().__init__()

        # geometry
        domain = num_exc  # Bounds of the grid, assuming left bottom corner is at (0, 0)
        exc_positions = GridPositions(domain)
        inh_positions = RandomPositions(domain)

        # neurons
        self.E = FNSNeuron(
            size=num_exc,
            C=0.25,
            g_L=16.7,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-60.0,
            tau_ref=4.0,
            V_initializer=bp.init.Uniform(-70.0, -50.0),
            method=method,
            embedding=exc_positions,
        )

        # Create a population of inhibitory neurons
        self.I = FNSNeuron(
            size=num_inh,
            C=0.25,
            g_L=16.7,
            V_L=-70.0,
            V_th=-50.0,
            V_rt=-60.0,
            tau_ref=4.0,
            V_initializer=bp.init.Uniform(-70.0, -50.0),
            method=method,
            embedding=inh_positions,
        )

        # Connectivity topology
        conn_E2E = DistanceDependent(
            GaussianKernel(
                sigma=3.0,
            ),
            self.E.embedding.domain,
            self.E.positions,
            boundary="periodic",
            include_self=False,
        )
        conn_E2I = bp.connect.FixedProb(prob=0.1)
        conn_I2E = bp.connect.FixedProb(prob=0.1)
        conn_I2I = DistanceDependent(
            GaussianKernel(
                sigma=3.0,
            ),
            self.I.embedding.domain,
            self.I.positions,
            boundary="periodic",
            include_self=False,
        )

        # Synapses
        self.E2E = Synapse(self.E, self.E, delay=2.0, conn=conn_E2E)
        self.E2I = Synapse(self.E, self.I, delay=2.0, conn=conn_E2I)
        self.I2E = Synapse(self.I, self.E, delay=2.0, conn=conn_I2E)
        self.I2I = Synapse(self.I, self.I, delay=2.0, conn=conn_I2I)

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)

    def to_dict(self):
        return {
            self.__class__.__name__: {
                "populations": {
                    "E": self.E.to_dict(),
                    "I": self.I.to_dict(),
                },
                "synapses": {
                    "E2E": self.E2E.to_dict(),
                    "E2I": self.E2I.to_dict(),
                    "I2E": self.I2E.to_dict(),
                    "I2I": self.I2I.to_dict(),
                },
            }
        }
