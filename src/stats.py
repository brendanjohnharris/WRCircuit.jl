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
from tqdm import trange

import json
import pickle

from brainpy import share
from typing import Union, Callable, Optional, Sequence, Any, Dict
from brainpy.types import ArrayType
from functools import partial


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
    def stats_run(swept_params):  # ! change this to NOT vmap, and vmap OUTSIDE
        results = run(swept_params)  # Dict of monitor outputs
        calc_stats = {key: jax.tree_map(func, results) for key, func in stats.items()}
        return calc_stats

    return stats_run


def save(file, d):
    with open(file, "wb") as f:
        pickle.dump(d, f)
    return file


def load(file):
    with open(file, "rb") as f:
        d = pickle.load(f)
    return d


def progress_vmap(
    func: callable,
    batch_size: int,
    clear_buffer: bool = True,
):
    def _vmap(arguments: Union[Dict[str, ArrayType], Sequence[ArrayType]]):
        if not isinstance(arguments, (dict, tuple, list)):
            raise TypeError(
                f'"arguments" must be sequence or dict, but we got {type(arguments)}'
            )

        # Convert to appropriate array type
        array_func = np.array if clear_buffer else jnp.array
        arguments = jax.tree.map(array_func, arguments)

        # Flatten the pytree
        flat_args, tree_def = jax.tree.flatten(arguments)

        # Verify uniform length
        lengths = [len(ele) for ele in flat_args]
        if len(np.unique(lengths)) != 1:
            raise ValueError(
                f"All elements in parameters should have the same length. "
                f"But we got {jax.tree.unflatten(tree_def, lengths)}"
            )

        n_samples = lengths[0]

        # Create vmapped function - define once
        vfunc = vmap(func)

        # Process in batches
        all_batch_results = []
        res_tree = None

        for i in trange(0, n_samples, batch_size):
            # Create batch
            batch_slice = [ele[i : i + batch_size] for ele in flat_args]
            batch_args = jax.tree.unflatten(tree_def, batch_slice)

            # Process batch
            batch_result = vfunc(batch_args)

            # Extract values and structure
            batch_values, batch_tree = jax.tree.flatten(
                batch_result, is_leaf=lambda a: isinstance(a, bm.Array)
            )

            # Store structure from first batch
            if res_tree is None:
                res_tree = batch_tree

            # Store batch results
            all_batch_results.append(batch_values)

        # Handle empty results
        if not all_batch_results:
            return None

        # Transpose to group by value index instead of batch
        n_values = len(all_batch_results[0])
        value_batches = [[] for _ in range(n_values)]

        for batch_values in all_batch_results:
            for i, val in enumerate(batch_values):
                value_batches[i].append(np.asarray(val) if clear_buffer else val)

        # Concatenate each value's batches
        concat_func = np.concatenate if clear_buffer else jnp.concatenate
        final_values = [concat_func(batches, axis=0) for batches in value_batches]

        # Clear buffer if requested
        if clear_buffer:
            bm.clear_buffer_memory()

        # Return reconstructed result
        return jax.tree.unflatten(res_tree, final_values), arguments

    return _vmap


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

        if spikes.dtype != bool:
            return jnp.full(seg_size, np.nan)

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


@jax.jit
def temporal_average(V):
    # # * If V is not an array of floats, return NaN
    # if (V.dtype != jnp.float32) or (V.dtype != jnp.float64):
    #     return jnp.full(V.shape[1], np.nan)  # Drop the time axis

    mean_V = jnp.mean(V, axis=0)
    return mean_V


def grand_distribution(nbins):
    @jax.jit
    def _grand_distribution(X):
        """
        Calculate the total distribution of a given variable X.
        """
        lower = jnp.float32(jnp.min(X))
        upper = jnp.float32(jnp.max(X))
        bin_size = (upper - lower) / nbins
        bin_edges = jnp.linspace(lower, upper, nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        indices = jnp.floor((X - lower) / bin_size).astype(int)
        indices = jnp.clip(indices, 0, nbins - 1)
        hist = jnp.bincount(indices.flatten(), length=nbins)
        return hist, bin_centers

    return _grand_distribution
