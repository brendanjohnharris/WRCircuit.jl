#### python
# filepath: /home/brendan/OneDrive/Masters/Code/Vortices/Julia/Dewdrop.jl/scripts/sbi_network_frequency.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
import brainpy as bp
import brainpy.math as bm
from scipy import signal
import sbijax
from sbijax import FMPE
from sbijax.nn import make_cnf
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.FNS import FNS

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
np.random.seed(42)
key = jax.random.PRNGKey(42)
bp.math.set_dt(0.05)
TARGET_FREQUENCY = 55.0
N_e = 4000
J_e = 0.0008
delta = 4.0
nu = 10
n_ext = 80


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_peak_frequency(lfp, dt=bp.share.load("dt")):
    fs = 1000.0 / dt
    freqs, power = signal.welch(lfp, fs, nperseg=min(8192, len(lfp)))
    mask = freqs > 1.0
    peak_idx = np.argmax(power[mask]) + np.sum(~mask)
    peak_freq = freqs[peak_idx]
    return peak_freq, power, freqs


def simulate_network(params, duration=10.0, discard_time=2.0, seed=None):
    """Now can handle the per-sample RNG from subkeys[i]."""
    # Create a jax PRNGKey if we have a seed
    if seed is None:
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    elif isinstance(seed, jnp.ndarray) and seed.shape == (2,):
        rng = seed
    else:
        rng = jax.random.PRNGKey(seed)
    print(rng)

    omega_ee = params["omega_ee"]
    omega_ie = params["omega_ie"]
    omega_ei = params["omega_ei"]
    omega_ii = params["omega_ii"]
    delta = params["delta"]

    model = FNS(
        N_e=4000,
        omega_ee=omega_ee,
        omega_ie=omega_ie,
        omega_ei=omega_ei,
        omega_ii=omega_ii,
        delta=delta,
        gamma=4,
        nu=10,
        n_ext=80,
        J_e=0.0008,
        method="exp_auto",
        key=rng,
    )
    monitors = ["E.spike"]
    runner = bp.DSRunner(model, monitors=monitors, numpy_mon_after_run=True)
    runner.run(duration=duration)
    discard_steps = int(discard_time / bp.math.get_dt())
    spike_monitor = runner.mon["E.spike"][:, discard_steps:]
    times = runner.mon["ts"][discard_steps:].view()
    lfp = bp.measure.unitary_LFP(times, spike_monitor.view())

    peak_freq, _, _ = calculate_peak_frequency(lfp)
    return peak_freq


# -----------------------------------------------------------------------------
# SBI-JAX FUNCTIONS
# -----------------------------------------------------------------------------
def prior_fn():
    """Return a 5D JointDistributionNamed for our 5 parameters."""
    return tfd.JointDistributionNamed(
        dict(
            omega_ee=tfd.Uniform(low=0.1, high=0.5),
            omega_ie=tfd.Uniform(low=0.2, high=0.7),
            omega_ei=tfd.Uniform(low=0.2, high=0.5),
            omega_ii=tfd.Uniform(low=0.3, high=0.8),
            delta=tfd.Uniform(low=2.0, high=6.0),
        ),
        validate_args=False,
    )


def simulator(theta, seed=None):
    """
    Called by sbijax with:
      - theta: a dict of parameter arrays (could be scalar or batched)
      - seed: a PRNG seed (integer), or None
    Returns a float or jnp.array of shape (batch_size,)
    """
    # We check whether it's a single sample or multiple samples:
    if theta["omega_ee"].ndim == 0:
        # ----- SINGLE SAMPLE -----
        params = {k: float(v) for k, v in theta.items()}
        return simulate_network(params, seed=seed)  # single float
    else:
        # ----- BATCHED SAMPLES -----
        num_sims = theta["omega_ee"].shape[0]
        if seed is None:
            main_key = jax.random.PRNGKey(np.random.randint(0, 2**32))
        elif isinstance(seed, jnp.ndarray) and seed.shape == (2,):
            main_key = seed  # treat this 2D array as the key
        else:
            main_key = jax.random.PRNGKey(seed)

        subkeys = jax.random.split(main_key, num_sims)

        sim_outputs = []
        for i in range(num_sims):
            params = {k: float(v[i]) for k, v in theta.items()}
            # pass subkeys[i], or subkeys[i][1], etc.  up to you
            sim_outputs.append(
                simulate_network(
                    params, seed=None if subkeys[i] is None else subkeys[i][1]
                )
            )
        return jnp.array(sim_outputs)  # shape (num_sims,)


def run_sbi(n_simulations=10, n_rounds=3):
    n_dim_theta = 5
    # Make a CNF network for density estimation
    neural_network = make_cnf(n_dim_theta)

    # Pass the prior function and simulator function to FMPE
    fns = (prior_fn, simulator)
    inference = FMPE(fns, neural_network)

    # Simulate initial data
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    data, _ = inference.simulate_data(subkey, n_simulations=n_simulations)

    # Convert the dictionary of thetas into array form for logging, if desired
    # (SBI-JAX already organizes data in "theta" and "x")
    # For example, if you want them for debugging:
    # theta_array = jnp.column_stack([
    #     data["theta"]["omega_ee"],
    #     data["theta"]["omega_ie"],
    #     data["theta"]["omega_ei"],
    #     data["theta"]["omega_ii"],
    #     data["theta"]["delta"],
    # ])
    # x_samples = jnp.array(data["x"]).reshape(-1, 1)

    # Fit the FMPE
    key, subkey = jax.random.split(key)
    fmpe_params, info = inference.fit(subkey, data=data)

    # Perform multiple rounds of inference
    for round_idx in range(1, n_rounds):
        print(f"Running round {round_idx + 1}...")
        key, subkey = jax.random.split(key)
        posterior_samples = inference.sample_posterior(
            subkey, fmpe_params, jnp.array([TARGET_FREQUENCY]), n_samples=n_simulations
        )
        print(f"Running {n_simulations} simulations for round {round_idx + 1}...")

        # Simulate new data from these posterior samples
        x_samples = []
        for i in tqdm(range(n_simulations)):
            key, subkey = jax.random.split(key)
            params = {
                "omega_ee": posterior_samples[i, 0],
                "omega_ie": posterior_samples[i, 1],
                "omega_ei": posterior_samples[i, 2],
                "omega_ii": posterior_samples[i, 3],
                "delta": posterior_samples[i, 4],
            }
            # Just call simulate_network directly here,
            # or call simulator(...) with the same param structure
            freq = simulate_network(params, seed=subkey[1])
            x_samples.append(freq)

        x_samples = jnp.array(x_samples).reshape(-1, 1)
        new_data = {
            "theta": posterior_samples,
            "x": x_samples,
        }

        # Fit again with the newly augmented data
        key, subkey = jax.random.split(key)
        fmpe_params, info = inference.fit(subkey, data=new_data, n_epochs=1000)

    return inference, fmpe_params


# -----------------------------------------------------------------------------
# PLOTTING UTILITIES
# -----------------------------------------------------------------------------
def plot_posterior_samples(inference, fmpe_params, n_samples=1000):
    key = jax.random.PRNGKey(0)
    theta_samples = inference.sample_posterior(
        key, fmpe_params, jnp.array([TARGET_FREQUENCY]), n_samples=n_samples
    )
    df = pd.DataFrame(
        {
            "omega_ee": theta_samples[:, 0],
            "omega_ie": theta_samples[:, 1],
            "omega_ei": theta_samples[:, 2],
            "omega_ii": theta_samples[:, 3],
            "delta": theta_samples[:, 4],
        }
    )
    plt.figure(figsize=(15, 15))
    sns.pairplot(df, corner=True)
    plt.suptitle(
        f"Posterior samples for target frequency {TARGET_FREQUENCY} Hz", y=1.02
    )
    plt.tight_layout()
    plt.savefig("posterior_pairplot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, param in enumerate(
        ["omega_ee", "omega_ie", "omega_ei", "omega_ii", "delta"]
    ):
        plt.subplot(1, 5, i + 1)
        sns.violinplot(y=df[param])
        plt.title(param)
    plt.tight_layout()
    plt.savefig("posterior_violins.png", dpi=300)
    plt.close()

    return df


def plot_training_trajectory(inference):
    losses = inference.training_history
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=300)
    plt.close()


def plot_frequency_vs_parameters(inference, fmpe_params, n_samples=100):
    key = jax.random.PRNGKey(0)
    theta_samples = inference.sample_posterior(
        key, fmpe_params, jnp.array([TARGET_FREQUENCY]), n_samples=n_samples
    )
    frequencies = []
    for i in tqdm(range(n_samples)):
        key, subkey = jax.random.split(key)
        params = {
            "omega_ee": theta_samples[i, 0],
            "omega_ie": theta_samples[i, 1],
            "omega_ei": theta_samples[i, 2],
            "omega_ii": theta_samples[i, 3],
            "delta": theta_samples[i, 4],
        }
        freq = simulate_network(params, seed=subkey[1])
        frequencies.append(freq)

    df = pd.DataFrame(
        {
            "omega_ee": theta_samples[:, 0],
            "omega_ie": theta_samples[:, 1],
            "omega_ei": theta_samples[:, 2],
            "omega_ii": theta_samples[:, 3],
            "delta": theta_samples[:, 4],
            "frequency": frequencies,
        }
    )

    plt.figure(figsize=(15, 10))
    for i, param in enumerate(
        ["omega_ee", "omega_ie", "omega_ei", "omega_ii", "delta"]
    ):
        plt.subplot(2, 3, i + 1)
        plt.scatter(df[param], df["frequency"], alpha=0.7)
        plt.xlabel(param)
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Frequency vs {param}")
        plt.axhline(y=TARGET_FREQUENCY, color="r", linestyle="--")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("frequency_vs_parameters.png", dpi=300)
    plt.close()

    return df


def plot_example_simulation(params, duration=500.0, discard_time=100.0):
    peak_freq = simulate_network(params, seed=None)
    omega_ee = params["omega_ee"]
    omega_ie = params["omega_ie"]
    omega_ei = params["omega_ei"]
    omega_ii = params["omega_ii"]
    delta = params["delta"]

    model = FNS(
        N_e=N_e,
        omega_ee=omega_ee,
        omega_ie=omega_ie,
        omega_ei=omega_ei,
        omega_ii=omega_ii,
        delta=delta,
        gamma=4,
        nu=nu,
        n_ext=n_ext,
        J_e=J_e,
        method="exp_auto",
        key=jax.random.PRNGKey(np.random.randint(0, 2**32)),
    )
    monitors = ["E.spike", "I.spike"]
    runner = bp.DSRunner(model, monitors=monitors, numpy_mon_after_run=True)
    runner.run(duration=duration)
    discard_steps = int(discard_time / bp.math.get_dt())

    e_spikes = runner.mon["E.spike"][:, discard_steps:]
    i_spikes = runner.mon["I.spike"][:, discard_steps:]
    times = runner.mon["ts"][discard_steps:].view()
    lfp = bp.measure.unitary_LFP(times, e_spikes.view())

    _, power, freqs = calculate_peak_frequency(lfp)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = np.arange(0, duration - discard_time, bp.math.get_dt())
    e_indices, e_times = np.where(e_spikes[:100])
    axes[0].scatter(time[e_times], e_indices, s=1, alpha=0.5)
    axes[0].set_ylabel("E Neuron Index")
    axes[0].set_title("Spike Raster Plot")

    i_indices, i_times = np.where(i_spikes)
    axes[1].scatter(time[i_times], i_indices, s=1, alpha=0.5)
    axes[1].set_ylabel("I Neuron Index")

    axes[2].plot(time, lfp)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("LFP")
    plt.tight_layout()
    plt.savefig("example_simulation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, power)
    plt.axvline(x=peak_freq, linestyle="--", label=f"Peak: {peak_freq:.2f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("power_spectrum.png", dpi=300)
    plt.close()

    return peak_freq


def main():
    print(f"Running SBI to find parameters for target frequency: {TARGET_FREQUENCY} Hz")
    inference, fmpe_params = run_sbi(n_simulations=500, n_rounds=3)

    # Plot posteriors
    posterior_samples = plot_posterior_samples(inference, fmpe_params)
    # Plot training loss
    plot_training_trajectory(inference)
    # Plot posterior-based frequency sampling
    freq_params_df = plot_frequency_vs_parameters(inference, fmpe_params)

    # Pick the sample whose frequency is closest to target
    closest_idx = np.argmin(np.abs(freq_params_df["frequency"] - TARGET_FREQUENCY))
    best_params = {
        "omega_ee": freq_params_df["omega_ee"].iloc[closest_idx],
        "omega_ie": freq_params_df["omega_ie"].iloc[closest_idx],
        "omega_ei": freq_params_df["omega_ei"].iloc[closest_idx],
        "omega_ii": freq_params_df["omega_ii"].iloc[closest_idx],
        "delta": freq_params_df["delta"].iloc[closest_idx],
    }
    print("\nBest parameters found (closest to 55 Hz):")
    for param, value in best_params.items():
        print(f"{param}: {value:.4f}")

    print("\nRunning example simulation with best parameters...")
    peak_freq = plot_example_simulation(best_params)
    print(f"Peak frequency: {peak_freq:.2f} Hz")
    print("\nAll plots have been saved to the current directory.")


if __name__ == "__main__":
    main()
