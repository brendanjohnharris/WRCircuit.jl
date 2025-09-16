#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.11 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
using LinearAlgebra
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin
    model = Dewdrop.models.Spatial
    begin # FNS parameters
        dx = 0.5
        rho = 20000.0
        kernel = Dewdrop.distances.ExponentialKernel
        delta = 4.0
        nu = 4.5 # External population firing rate
        n_ext = 100  # Number of external synapses per Exc. neuron

        sigma_ee = 0.075  # Width of the distance-dependent connectivity kernel (mm)
        sigma_ei = 0.095
        sigma_ie = 0.19
        sigma_ii = 0.19

        K_ee = 270
        K_ei = 350
        K_ie = 165
        K_ii = 200
    end
end

begin
    tmax = 50u"s" # * Bump up
    tmin = 10u"s" # The transient. Simulations always begin at 0
    fixed_params = (; dx, rho, kernel, n_ext,
                    sigma_ee, sigma_ei, sigma_ie, sigma_ii,
                    K_ee, K_ie, K_ei, K_ii, nu)

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, 2000; populations = [:E, :I], vars = [:spike, :V, :input])
end

# * Check we are near the critical point
begin # * Now pull out membrane potential time series
    V = x[Population = At(:E), Var = At(:input)]
    lines(V[:, 1])
end
