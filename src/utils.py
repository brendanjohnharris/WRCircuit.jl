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


def create_run(
    model, fixed_params, monitors, duration, transient=0.0, concrete_out=False
):

    transient_idx = int(transient / bp.share["dt"])

    @jax.jit
    def run(swept_params):
        m = model(**fixed_params, **swept_params)
        runner = bp.DSRunner(m, monitors=monitors, numpy_mon_after_run=concrete_out)
        runner.run(duration=duration)
        return {m: runner.mon[m][transient_idx:, :] for m in monitors}

    return run


def create_stats_run(run, stats):
    """
    run is a create_run function returning a given set of monitors.
    stats is a Dict of statname=>function pairs. Each function should accept the dict
    of create_run as an input, select the relevant monitor, and return the desired
    statistic. Assuems you are using the bp.share["dt"]
    """

    @jax.jit
    def stats_run(swept_params):
        results = jax.vmap(run, in_axes=0)(swept_params)  # Dict of monitor outputs
        calcs = {
            key: jax.vmap(lambda x: jax.tree_map(func, x), in_axes=0)
            for key, func in stats.items()
        }  # Dict of vmapped funcs, vmapped over first dim (batch)
        calc_stats = {
            key: func(results) for key, func in calcs.items()
        }  # Dict of stat outputs
        return calc_stats

    return stats_run


@partial(jax.jit, static_argnames=["bin_size", "axis"])
def coarsegrain(spikes, bin_size, axis=0):
    # Move the specified axis to the front
    spikes = jnp.moveaxis(spikes, axis, 0)
    n_steps = spikes.shape[0]
    n_bins = n_steps // bin_size
    truncated_spikes = spikes[: n_bins * bin_size]
    reshaped_spikes = jnp.reshape(truncated_spikes, (n_bins, bin_size, -1))
    binned_spikes = jnp.sum(reshaped_spikes, axis=1)
    # Move the axis back to its original position
    binned_spikes = jnp.moveaxis(binned_spikes, 0, axis)
    return binned_spikes


# * Some stats functions
@jax.jit
def firing_rate(spikes, dt=bp.share["dt"]):
    """
    Calculate the firing rate of a neuron given its spike train. If 'spikes' is not a boolean array, return an array of nans. Should be jax compatible.
    """
    # * If spikes is not a boolean array, return an array of nans
    if spikes.dtype != bool:
        return jnp.full(spikes.shape[1], np.nan)  # Drop the time axis

    total_spikes = jnp.sum(
        spikes, axis=0
    )  # The first axis is the batch, the second axis is the time

    total_time = spikes.shape[0] * (dt / 1000.0)
    mean_rate = total_spikes / total_time
    return mean_rate


def susceptibility(bin, dt=bp.share["dt"]):
    @jax.jit
    def _susceptibility(spikes):
        # * If spikes is not a boolean array, return an array of nans
        if spikes.dtype != bool:
            return jnp.nan

        # * Bin over time
        if bin is not None:
            bin_size = int(bin / dt)
            spikes = coarsegrain(spikes, bin_size, axis=0)  # Coarse grain over time

        active_neurons = spikes > 0

        # * Fraction of active neurons at each time step
        rho = jnp.mean(active_neurons, axis=1)  # mean across neurons

        # * Susceptibility
        chi = jnp.mean(rho**2) - jnp.mean(rho) ** 2
        return chi

    return _susceptibility


def spike_spectrum(n_segments=1):
    @jax.jit
    def _spike_spectrum(spikes):
        """
        Compute the Bartlett spectrum for each neuron (column) of `data`
        and then average across neurons.

        Args:
            data: A (T, N) array of time-series data, one column per neuron.
                E.g., discrete spike counts or continuous signals.
            n_segments: Number of segments to split the time axis into
                        for Bartlett's method. Must be treated as static
                        for JIT compilation.
            dt: Sampling interval (if you need frequency in Hz, etc.).

        Returns:
            A 1D array of length `seg_size`, representing the Bartlett-averaged
            power spectral density (PSD), further averaged across all neurons.
            Frequencies can be obtained separately with jnp.fft.fftfreq(seg_size, d=dt).
        """

        T, N = spikes.shape
        seg_size = T // n_segments  # integer division; remainder is discarded if any

        # Reshape data into (n_segments, seg_size, N)
        # so each segment is [segment_index, time_in_segment, neuron].
        data_reshaped = spikes[: seg_size * n_segments, :].reshape(
            n_segments, seg_size, N
        )

        # Compute the FFT of each segment along the time-in-segment axis (axis=1).
        # Shape: (n_segments, seg_size, N)
        fft_segments = jnp.fft.fft(data_reshaped, axis=1)

        # Periodogram for each segment/neuron: (1/seg_size)*|FFT|^2.
        # Still shape: (n_segments, seg_size, N)
        periodograms = (1.0 / seg_size) * jnp.abs(fft_segments) ** 2

        # Average over segments -> shape (seg_size, N)
        psd_per_neuron = jnp.mean(periodograms, axis=0)

        # Now average over neurons -> shape (seg_size,)
        psd_avg = jnp.mean(psd_per_neuron, axis=1)

        return psd_avg

    return _spike_spectrum
