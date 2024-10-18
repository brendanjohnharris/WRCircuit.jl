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


class Positions(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def cast_to_tuple(self, x):
        if isinstance(x, (list, np.ndarray)):
            return tuple(x)
        elif isinstance(x, tuple):
            return x
        else:  # For scalars, wrap it in a tuple
            return (x,)


class GridPositions(Positions):
    def __init__(self, size):
        self.size = self.cast_to_tuple(size)

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        if len(shape) != len(self.size):
            raise ValueError("Shape and size must have the same length")
        grids = []
        for s, n in zip(self.size, shape):
            offset = (s / n) / 2  # Offset to center the grid
            grids.append(np.linspace(0 + offset, s + offset, n, endpoint=False))
        positions = list(product(*grids))
        return positions


class RandomPositions(Positions):
    def __init__(self, size):
        self.size = self.cast_to_tuple(size)

    def __call__(self, shape):
        shape = self.cast_to_tuple(shape)
        total_positions = np.prod(shape)
        positions = []
        for s in self.size:
            positions.append(np.random.uniform(0, s, total_positions))
        positions = list(zip(*positions))
        return positions


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


class FNSNeuron(GradNeuDyn):
    r"""
    Treves 1993 neuron model with adaptation current.

    **Model Description**

    The neuron model follows the equations described in Treves (1993), including a potassium
    current for adaptation in excitatory neurons.

    **Membrane Potential Dynamics**

    .. math::

        C \frac{dV}{dt} = -g_L (V - V_L) + I_K + I_{\text{ext}}

    where:

    - \( V \) is the membrane potential.
    - \( C \) is the membrane capacitance.
    - \( g_L \) is the leak conductance.
    - \( V_L \) is the leak reversal potential.
    - \( I_K \) is the potassium adaptation current.
    - \( I_{\text{ext}} \) is the external input current.

    **Potassium Current**

    .. math::

        I_K = -g_K (V - V_K)

    where:

    - \( g_K \) is the potassium conductance.
    - \( V_K \) is the potassium reversal potential.

    **Potassium Conductance Dynamics**

    .. math::

        \frac{d g_K}{dt} = -\frac{g_K}{\tau_K} + \Delta g_K \sum_k \delta(t - t_k)

    - \( \tau_K \) is the adaptation time constant.
    - \( \Delta g_K \) is the conductance increment upon each spike.
    - \( t_k \) are the spike times.

    **Spike Dynamics**

    - When \( V \geq V_{\text{th}} \), the neuron emits a spike.
    - The membrane potential is reset to \( V_{\text{rt}} \).
    - The neuron enters a refractory period of duration \( \tau_{\text{ref}} \).

    **Parameters**

    ============= =============== ======= ==========================================
    **Parameter** **Default**     **Unit** **Description**
    ------------- --------------- ------- ------------------------------------------
    C             250             pF      Membrane capacitance.
    g_L           16.7            nS      Leak conductance.
    V_L           -70.0           mV      Leak reversal potential.
    V_K           -85.0           mV      Potassium reversal potential.
    V_th          -50.0           mV      Spike threshold.
    V_rt          -60.0           mV      Reset potential.
    tau_ref       4.0             ms      Refractory period duration.
    tau_K         80.0            ms      Adaptation time constant.
    Delta_g_K     10.0            nS      Conductance increment upon spike.
    ============= =============== ======= ==========================================

    **Variables**

    =================== ================= ===================================================
    **Variable Name**   **Initial Value**  **Description**
    ------------------- ------------------ ---------------------------------------------------
    V                   V_initializer     Membrane potential.
    g_K                 g_K_initializer   Potassium conductance.
    spike               False             Spike indicator.
    t_last_spike        -1e8              Time of the last spike.
    =================== ================= ===================================================

    **Example Usage**

    .. code-block:: python

        import brainpy as bp

        # Define neuron parameters
        neuron = FNSNeuron(size=100, Delta_g_K=10.0)

        # Run the simulation
        runner = bp.DSRunner(neuron, monitors=['V', 'spike'])
        runner.run(100.0)

        # Plot results
        bp.visualize.line_plot(runner.mon.ts, runner.mon.V, show=True)

    """

    def __init__(
        self,
        size: Shape,
        sharding: Optional[Sequence[str]] = None,
        keep_size: bool = False,
        mode: Optional[bm.Mode] = None,
        name: Optional[str] = None,
        spk_fun: Callable = bm.surrogate.InvSquareGrad(),
        spk_dtype: Any = None,
        spk_reset: str = "hard",
        detach_spk: bool = False,
        method: str = "exp_auto",
        init_var: bool = True,
        scaling: Optional[bm.Scaling] = None,
        # neuron parameters
        C: Union[float, ArrayType, Callable] = 0.25,  # nF
        g_L: Union[float, ArrayType, Callable] = 16.7,  # nS
        V_L: Union[float, ArrayType, Callable] = -70.0,  # mV
        V_K: Union[float, ArrayType, Callable] = -85.0,  # mV
        V_th: Union[float, ArrayType, Callable] = -50.0,  # mV
        V_rt: Union[float, ArrayType, Callable] = -60.0,  # mV
        tau_ref: Union[float, ArrayType, Callable] = 4.0,  # ms
        tau_K: Union[float, ArrayType, Callable] = 80.0,  # ms
        Delta_g_K: Union[float, ArrayType, Callable] = 10.0,  # nS
        positions: Union[None, ArrayType, Callable] = None,
        V_initializer: Union[Callable, ArrayType] = ZeroInit(),
        g_K_initializer: Union[Callable, ArrayType] = ZeroInit(),
        # noise
        noise: Union[float, ArrayType, Callable] = None,
    ):
        super().__init__(
            size=size,
            name=name,
            keep_size=keep_size,
            mode=mode,
            sharding=sharding,
            spk_fun=spk_fun,
            detach_spk=detach_spk,
            method=method,
            spk_dtype=spk_dtype,
            spk_reset=spk_reset,
            scaling=scaling,
        )

        # parameters
        self.C = self.init_param(C)
        self.g_L = self.init_param(g_L)
        self.V_L = self.init_param(V_L)
        self.V_K = self.init_param(V_K)
        self.V_th = self.init_param(V_th)
        self.V_rt = self.init_param(V_rt)
        self.tau_ref = self.init_param(tau_ref)
        self.tau_K = self.init_param(tau_K)
        self.Delta_g_K = self.init_param(Delta_g_K)

        if positions is None:
            positions = GridPositions(self.size)
        if callable(positions):
            positions = positions(self.size)  # Initialize here

        self.positions = positions

        # initializers
        self._V_initializer = is_initializer(V_initializer)
        self._g_K_initializer = is_initializer(g_K_initializer)

        # integral
        self.noise = init_noise(noise, self.varshape, num_vars=2)
        if self.noise is not None:
            self.integral = sdeint(method=self.method, f=self.derivative, g=self.noise)
        else:
            self.integral = odeint(method=method, f=self.derivative)

        # variables
        if init_var:
            self.reset_state(self.mode)

    def dV(self, V, t, g_K, I_ext):
        I_ext = self.sum_current_inputs(V, init=I_ext)
        I_K = -g_K * (V - self.V_K)
        dVdt = (-self.g_L * (V - self.V_L) + I_K + I_ext) / self.C
        return dVdt

    def dg_K(self, g_K, t):
        dg_Kdt = -g_K / self.tau_K
        return dg_Kdt

    @property
    def derivative(self):
        return JointEq([self.dV, self.dg_K])

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.init_variable(self._V_initializer, batch_size)
        self.g_K = self.init_variable(self._g_K_initializer, batch_size)
        self.spike = self.init_variable(
            partial(bm.zeros, dtype=self.spk_dtype), batch_size
        )
        self.t_last_spike = self.init_variable(bm.ones, batch_size)
        self.t_last_spike.fill_(-1e8)

    def update(self, I_ext=None):
        t = share.load("t")
        dt = share.load("dt")
        if I_ext is None:
            I_ext = 0.0

        # integrate variables
        V, g_K = self.integral(self.V.value, self.g_K.value, t, I_ext, dt)
        V += self.sum_delta_inputs()

        # refractory period
        refractory = (t - self.t_last_spike) <= self.tau_ref
        V = bm.where(refractory, self.V.value, V)

        # spike
        if isinstance(self.mode, bm.TrainingMode):
            spike = self.spk_fun(V - self.V_th)
            spike_no_grad = stop_gradient(spike) if self.detach_spk else spike

            if self.spk_reset == "soft":
                V -= (self.V_th - self.V_rt) * spike_no_grad
            elif self.spk_reset == "hard":
                V += (self.V_rt - V) * spike_no_grad
            else:
                raise ValueError

            t_last_spike = stop_gradient(
                bm.where(spike_no_grad > 0.0, t, self.t_last_spike.value)
            )
            refractory = stop_gradient(
                bm.logical_or(refractory, spike_no_grad > 0.0).value
            )

            # Update g_K upon spike
            g_K += self.Delta_g_K * spike_no_grad

        else:
            spike = V >= self.V_th
            V = bm.where(spike, self.V_rt, V)
            t_last_spike = bm.where(spike, t, self.t_last_spike.value)

            # Update g_K upon spike
            g_K = bm.where(spike, g_K + self.Delta_g_K, g_K)

        # update variables
        self.V.value = V
        self.g_K.value = g_K
        self.spike.value = spike
        self.t_last_spike.value = t_last_spike

        return spike


class AMPASyn(bp.Projection):
    def __init__(
        self, pre, post, delay, conn, g_max=1.0, tau_d=5, tau_r=1, V_rev=0.0, alpha=1.0
    ):
        super().__init__()
        self.proj = bp.dyn.FullProjAlignPreSDMg(
            pre=pre,
            delay=delay,
            syn=bp.dyn.AMPA.desc(
                pre.num, alpha=alpha, beta=1 / tau_d, T=1 / tau_r, T_dur=tau_r
            ),
            comm=bp.dnn.CSRLinear(conn(pre_size=pre.size, post_size=post.size), g_max),
            out=bp.dyn.COBA(E=V_rev),
            post=post,
        )


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
        domain,
        positions_pre,
        positions_post=None,
        boundary="periodic",
        distance_metric=euclidean_distance,
        kernel=None,
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
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 2**32)
        self.key = jax.random.PRNGKey(self.seed)

    def build_mat(self):
        # Capture constants to pass into JIT-compiled functions
        boundary_code = self.boundary_code
        domain = self.domain
        distance_metric = self.distance_metric
        kernel = self.kernel
        kernel_params = getattr(self, "kernel_params", {})

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
            prob = kernel(dist, **kernel_params)
            return prob

        # Fix kernel and distance_metric using partial to ensure they are static
        compute_probability_fixed = partial(
            compute_probability, kernel=kernel, distance_metric=distance_metric
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


class GaussianKernel(DistanceDependent):
    """
    Distance-dependent connector with a Gaussian kernel.

    Parameters
    ----------
    pre_positions : array-like
        Positions of the pre-synaptic neurons.
    post_positions : array-like, optional
        Positions of the post-synaptic neurons. If not provided, defaults to pre_positions.
    sigma : float, keyword-only
        Width of the Gaussian.
    max_prob : float, optional
        Maximum probability of connection.
    **kwargs : dict
        Additional arguments passed to DistanceDependent.
    """

    def __init__(
        self,
        domain,
        pre_positions,
        post_positions=None,
        *,
        sigma,
        max_prob=1.0,
        **kwargs,
    ):
        self.sigma = sigma
        self.max_prob = max_prob

        # Define the Gaussian kernel function
        @jit
        def gaussian_kernel(distance, max_prob, sigma):
            return max_prob * jnp.exp(-(distance**2) / (2 * sigma**2))

        self.kernel_params = {"max_prob": max_prob, "sigma": sigma}

        super(GaussianKernel, self).__init__(
            domain=domain,
            positions_pre=pre_positions,
            positions_post=post_positions,
            kernel=gaussian_kernel,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, max_prob={self.max_prob}, "
            f"boundary={self.boundary}, distance_metric={self.distance_metric}, "
            f"include_self={self.include_self}, seed={self.seed})"
        )


class FNSCircuit(bp.DynSysGroup):
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
            positions=exc_positions,
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
            positions=inh_positions,
        )

        # Connectivity topology
        conn_E2E = GaussianKernel(
            domain,
            self.E.positions,
            sigma=5.0,
            boundary="absorbing",
            include_self=False,
        )
        conn_E2I = bp.connect.FixedProb(prob=0.0)
        conn_I2E = bp.connect.FixedProb(prob=0.0)
        conn_I2I = GaussianKernel(
            domain,
            self.I.positions,
            sigma=5.0,
            boundary="absorbing",
            include_self=False,
        )

        # Synapses
        self.E2E = AMPASyn(self.E, self.E, delay=10.0, conn=conn_E2E)
        self.E2I = AMPASyn(self.E, self.I, delay=10.0, conn=conn_E2I)
        self.I2E = AMPASyn(self.I, self.E, delay=10.0, conn=conn_I2E)
        self.I2I = AMPASyn(self.I, self.I, delay=10.0, conn=conn_I2I)

        # define input variables given to E/I populations
        self.Ein = bp.dyn.InputVar(self.E.varshape)
        self.Iin = bp.dyn.InputVar(self.I.varshape)
        self.E.add_inp_fun("", self.Ein)
        self.I.add_inp_fun("", self.Iin)
