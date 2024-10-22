import brainpy as bp
import brainpy.math as bm
import numpy as np
from scipy.special import gamma
from functools import partial
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt


# Define the neuron model equivalent to NEST's iaf_psc_delta with V_min parameter
class IafPscDeltaNeuron(bp.dyn.NeuDyn):
    def __init__(
        self,
        size,
        C_m=1.0,
        tau_m=20.0,
        t_ref=2.0,
        E_L=0.0,
        V_reset=10.0,
        V_th=20.0,
        V_min=-10.0,
        method="exp_auto",
        **kwargs
    ):
        super().__init__(size=size, **kwargs)
        self.C_m = C_m
        self.tau_m = tau_m
        self.t_ref = t_ref
        self.E_L = E_L
        self.V_reset = V_reset
        self.V_th = V_th
        self.V_min = V_min

        # Initialize variables
        self.V = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # ODE integrator
        self.integral = bp.odeint(method=method, f=self.derivative)

    def derivative(self, V, t, I_syn):
        dVdt = (-(V - self.E_L) + I_syn) / (self.tau_m / self.C_m)
        return dVdt

    def update(self, tdi):
        t, dt = tdi["t"], tdi["dt"]
        I_syn = self.get_input("input") + self.sum_input("input_post")
        V = self.integral(self.V.value, t, I_syn, dt)
        # Enforce V >= V_min
        V = bm.maximum(V, self.V_min)
        # Refractory period
        refractory = (t - self.t_last_spike.value) <= self.t_ref
        V = bm.where(refractory, self.V.value, V)
        # Spike generation
        self.spike.value = bm.logical_and(V >= self.V_th, ~refractory)
        self.t_last_spike.value = bm.where(self.spike.value, t, self.t_last_spike.value)
        # Reset membrane potential
        V = bm.where(self.spike.value, self.V_reset, V)
        self.V.value = V


# Define the network
class HeterogeneousCircuit(bp.DynSysGroup):
    def __init__(
        self,
        dt=0.1,
        delay=1.5,
        epsilon=0.1,
        tauMem=20.0,
        theta=20.0,
        Vr=10.0,
        g=4.0,
        eta=0.3,
        orderCE=4000,
        J=0.02,
        alpha=2.0,
        simtime=10000.0,
        **kwargs
    ):
        super().__init__()

        # Parameters
        self.dt = dt
        self.delay = delay
        self.epsilon = epsilon
        self.tauMem = tauMem
        self.theta = theta
        self.Vr = Vr
        self.g = g
        self.eta = eta
        self.orderCE = orderCE
        self.J = J
        self.alpha = alpha
        self.simtime = simtime

        # Calculate order, NE, NI
        order = int(self.orderCE / (self.epsilon * 4))
        NE = 4 * order
        NI = 1 * order
        self.NE = NE
        self.NI = NI
        self.N_neurons = NE + NI

        # CE and CI
        self.CE = int(self.epsilon * NE)
        self.CI = int(self.epsilon * NI)

        # Neuron parameters
        self.J_ex = self.J
        self.J_in = -self.g * self.J_ex
        self.V_reset = self.Vr
        self.V_th = self.theta
        self.V_min = -10.0  # as specified

        # Input parameters
        self.nu_th = self.theta / (self.J * self.CE * self.tauMem)
        self.nu_ex = self.eta * self.nu_th
        self.constant_input_amplitude = (
            self.CE * self.nu_ex * 10
        )  # as per the modification

        # Create neuron populations
        self.E = IafPscDeltaNeuron(
            size=self.NE,
            C_m=1.0,
            tau_m=self.tauMem,
            t_ref=2.0,
            E_L=0.0,
            V_reset=self.V_reset,
            V_th=self.V_th,
            V_min=self.V_min,
            method="exp_auto",
            name="Excitatory",
        )
        self.I = IafPscDeltaNeuron(
            size=self.NI,
            C_m=1.0,
            tau_m=self.tauMem,
            t_ref=2.0,
            E_L=0.0,
            V_reset=self.V_reset,
            V_th=self.V_th,
            V_min=self.V_min,
            method="exp_auto",
            name="Inhibitory",
        )

        # Generate synaptic weights
        self.generate_synaptic_weights()

        # Create synapses
        self.create_synapses()

        # Add constant input to all neurons
        self.add_constant_input()

    def generate_synaptic_weights(self):
        if self.alpha != 2.0:
            # Generate excitatory weights using Pareto distribution
            A_alpha = gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) / np.pi
            D = 0.5
            x1fac = (2 * A_alpha * D / self.alpha) ** (1 / self.alpha)
            x0fac = 1 - x1fac * self.alpha / (self.alpha - 1)
            # Excitatory weights
            rng = np.random.default_rng()
            samples_ex = (
                x1fac * (rng.pareto(self.alpha, (self.N_neurons, self.CE)) + 1) + x0fac
            )
            self.J_ex_tot = self.J_ex * samples_ex
            # Inhibitory weights, tight balance
            samples_in = []
            for sum_a in np.sum(samples_ex, axis=1):
                # Adjust inhibitory weights to maintain balance
                b = x1fac * (rng.pareto(self.alpha, self.CI) + 1) + x0fac
                samples_in.append(b)
            samples_in = np.array(samples_in)
            self.J_in_tot = self.J_in * samples_in
        else:
            # Classical model with fixed weights
            self.J_ex_tot = self.J_ex
            self.J_in_tot = self.J_in

    def create_synapses(self):
        # Connections from E to E and I
        conn_E = bp.connect.FixedIndegree(indegree=self.CE)
        if self.alpha != 2.0:
            # Use variable weights
            syn_EE = bp.dyn.ProjAlignPreMg(
                pre=self.E,
                post=self.E,
                conn=conn_E,
                weight=self.J_ex_tot[: self.NE],
                delay=self.delay,
            )
            syn_EI = bp.dyn.ProjAlignPreMg(
                pre=self.E,
                post=self.I,
                conn=conn_E,
                weight=self.J_ex_tot[self.NE :],
                delay=self.delay,
            )
        else:
            # Use fixed weights
            syn_EE = bp.dyn.ProjAlignFixed(
                pre=self.E, post=self.E, conn=conn_E, weight=self.J_ex, delay=self.delay
            )
            syn_EI = bp.dyn.ProjAlignFixed(
                pre=self.E, post=self.I, conn=conn_E, weight=self.J_ex, delay=self.delay
            )

        # Connections from I to E and I
        conn_I = bp.connect.FixedIndegree(indegree=self.CI)
        if self.alpha != 2.0:
            syn_IE = bp.dyn.ProjAlignPreMg(
                pre=self.I,
                post=self.E,
                conn=conn_I,
                weight=self.J_in_tot[: self.NE],
                delay=self.delay,
            )
            syn_II = bp.dyn.ProjAlignPreMg(
                pre=self.I,
                post=self.I,
                conn=conn_I,
                weight=self.J_in_tot[self.NE :],
                delay=self.delay,
            )
        else:
            syn_IE = bp.dyn.ProjAlignFixed(
                pre=self.I, post=self.E, conn=conn_I, weight=self.J_in, delay=self.delay
            )
            syn_II = bp.dyn.ProjAlignFixed(
                pre=self.I, post=self.I, conn=conn_I, weight=self.J_in, delay=self.delay
            )

        # Add synapses to the network
        self.EE_syn = syn_EE
        self.EI_syn = syn_EI
        self.IE_syn = syn_IE
        self.II_syn = syn_II

    def add_constant_input(self):
        # Create constant input
        self.Ein = bp.inputs.ConstantInput(self.E.num, self.constant_input_amplitude)
        self.Iin = bp.inputs.ConstantInput(self.I.num, self.constant_input_amplitude)
        # Add input to neurons
        self.E.input = self.Ein.current
        self.I.input = self.Iin.current

    def update(self, tdi):
        self.E.update(tdi)
        self.I.update(tdi)
        self.EE_syn.update(tdi)
        self.EI_syn.update(tdi)
        self.IE_syn.update(tdi)
        self.II_syn.update(tdi)
        self.Ein.update(tdi)
        self.Iin.update(tdi)
