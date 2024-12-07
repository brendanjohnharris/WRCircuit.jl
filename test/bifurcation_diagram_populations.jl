#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using PythonCall
using Dewdrop
using Unitful
using Statistics
using TimeseriesTools
using CairoMakie
using LinearAlgebra

model = models.population_model.FNSPopulations

begin # * Fixed parameters
    N = 2500
    T = 1000.0
end

begin
    x = bpsolve(model(N), T; populations = [:E, :I], vars = [:spike])
end

begin # * Order parameters
    const UnivariateSpikeTrain = Base.typeintersect(SpikeTrain, UnivariateTimeSeries)
    const MultivariateSpikeTrain = Base.typeintersect(SpikeTrain, MultivariateTimeSeries)
    function firingrate(x::UnivariateSpikeTrain)
        λ = sum(x) / duration(x)
        uconvert(unit(eltype(x)) * u"Hz", λ)
    end
    function firingrate(X::MultivariateSpikeTrain)
        firingrate.(eachslice(X, dims = dims(X)[2:end]))
    end
    function cv(x::UnivariateSpikeTrain)
        ts = times(x[x])
        isis = diff(ts)
        if length(isis) < 2
            return 0.0 # No spikes, mean isi is Inf, say cv is 0
        else
            return std(isis) / mean(isis)
        end
    end
    function cv(X::MultivariateSpikeTrain)
        cv.(eachslice(X, dims = dims(X)[2:end]))
    end
end

begin # * Sweep parameters
    order_parameter = cv
    population = :E

    ν̂s = range(0, 4, length = 2) |> Dim{:ν̂}
    gs = range(0, 8, length = 2) |> Dim{:g}
    params = ToolsArray(Iterators.product(N, ν̂s, gs) |> collect, (gs, ν̂s))
    X = map(params) do (g, ν̂)
        jax.clear_caches()
        PythonCall.GIL.lock(GC.gc)
        PythonCall.GC.gc() # Check these work with xla_bridge.get_backend().live_arrays()
        m = model(N; g, nu_hat = ν̂)
        x = bpsolve(m, T; populations = [:E, :I], vars = [:spike])
        mean(order_parameter(x[Population = At(population), Var = At(:spike)]))
    end
end

begin # * Plot bifurcation diagram
    f = Figure()
    ax = Axis(f[1, 1])
    heatmap!(ax, X)
    f
end

begin # * Particle swarm to resolve areas of detail
    using Optim
    using DifferentiationInterface
    using FiniteDiff

    function solve_pb(params)
        tanh(params[1] - params[2])
    end
    global trace = []
    function trace_pb(params)
        x = solve_pb(params)
        push!(trace, (params..., x))
        x
    end
    xs = -2:0.01:2
    ys = -2:0.01:2
    zs = [solve_pb([x, y]) for x in xs, y in ys]
    surface(xs, ys, zs)

    xs = [-1, 1]
    ys = [-1, 1]
    FiniteDiff.finite_difference_jacobian(solve_pb, [-1.0, -1.0])
    trace = []
    function gradnorm(x)
        -norm(FiniteDiff.finite_difference_jacobian(trace_pb, x))
    end
    # optimize(func, [-1.0, -1.0], ParticleSwarm(; n_particles = 3))
    o = optimize(gradnorm, [-0.9, 1.0],
                 ParticleSwarm(; lower = [-2.0, -2.0], upper = [2.0, 2.0], n_particles = 4))
    # ......... finite difference gradient func.......
    # ........optimize for maximum gradient.............
    scatter!(Point3f.(trace))
    current_figure()
end
