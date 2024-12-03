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

    ν̂s = range(3, 4, length = 2) |> Dim{:ν̂}
    gs = range(7, 8, length = 2) |> Dim{:g}
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
