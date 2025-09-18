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
        rho = 20000
        dx = 0.5
        sigma_ee = 0.075
        sigma_ei = 0.095
        sigma_ie = 0.19
        sigma_ii = 0.19
        gamma = 4
        K_ee = 270 # 270
        K_ei = 350
        K_ie = 165
        K_ii = 200
        delta = 3.0
        nu = 5.0
        n_ext = 100
        J_ee = 0.0010 # 0.00105
        J_ei = 0.00145
        tau_r_i = 2.0
        tau_d_i = 5.0
    end
end

begin
    tmax = 10u"s" # * Bump up
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (; rho,
                    dx,
                    sigma_ee,
                    sigma_ei,
                    sigma_ie,
                    sigma_ii,
                    gamma,
                    K_ee,
                    K_ei,
                    K_ie,
                    K_ii,
                    delta,
                    nu,
                    n_ext,
                    J_ee, # 0.00105
                    J_ei,
                    tau_r_i,
                    tau_d_i)

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, tmax; populations = [:E, :I], vars = [:spike, :V, :input],
                transient = tmin)
end

begin # * Animate
    spikes = x[Population = At(:E), Var = At(:spike)]
    dt = 50.0u"ms"
    function group_dt(x::T, dt::T) where {T}
        round(x / dt) * dt
    end
    rates = groupby(spikes, 𝑡 => Base.Fix2(group_dt, dt))
    rates = map(rates) do r
        dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", dt)
    end |> stack
    rates = permutedims(rates, (𝑡, Neuron))
    rates = rectify(rates, dims = 𝑡)

    function infer_geometry(A, dx)
        ns = lookup(A, Neuron)
        n = sqrt(length(ns)) |> Int
        ns = Dewdrop.python_reshape(ns, n, n)
        δx = dx / n
        x = range(δx / 2, dx, step = δx)
        idxs = ToolsArray(Matrix{eltype(ns)}(undef, n, n),
                          (DimensionalData.X(x), DimensionalData.Y(x)))
        idxs .= ns # Assume Neurons are just flattened
        positions = Iterators.product(x, x)
        positions = ToolsArray(collect(positions)[:], dims(A, Neuron))
        return positions, idxs
    end

    positions, idxs = infer_geometry(spikes, dx)
end

begin # * Animate
    color = Observable(zeros(length(positions)))
    colormap = cgrad([:transparent, :crimson])

    f = Figure()
    ax = Axis(f[1, 1])
    p = scatter!(ax, Point2f.(positions); markersize = 5, color, colormap,
                 colorrange = (0, ustrip(maximum(rates))))
    Colorbar(f[1, 2], p; label = "Firing rate (Hz)")
    record(f, "critical_demo.mp4", lookup(rates, 𝑡), framerate = 12) do t
        r = rates[𝑡 = At(t)] |> ustrip
        color[] = r
    end
end

begin # * Power spectrum of rate fluctuations
    V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
    V = set(V, 𝑡 => uconvert.(u"s", times(V)))
    V = rectify(V, dims = 𝑡)
    lines(V[:, 1]) |> display
    s = spectrum(V)
    s = mean(s, dims = Neuron)
    s = dropdims(s, dims = Neuron)[5:end]
    lines(s |> ustripall; axis = (; xscale = log10, yscale = log10))
end
