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
    N = 12500
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
        isempty(isis) && return 0.0 # No spikes, mean isi is Inf, say cv is 0
        std(isis) / mean(isis)
    end
    function cv(X::MultivariateSpikeTrain)
        cv.(eachslice(X, dims = dims(X)[2:end]))
    end
end

begin # * Sweep parameters
    order_parameter = cv
    population = :E

    ν̂s = range(0, 4, length = 3) |> Dim{:ν̂}
    gs = range(0, 8, length = 3) |> Dim{:g}
    params = ToolsArray(Iterators.product(N, ν̂s, gs) |> collect, (gs, ν̂s))
    # params = map(params) do (N, ν̂, g)
    #     (; N = N, nu_hat = ν̂, g = g)
    # end
    # params = params[:]
    # params = map(keys(first(params))) do k
    #     k => getindex.(params, [k])
    # end |> Dict
    X = map(params) do (g, ν̂)
        # PythonCall.GC.gc()
        x = bpsolve(model(N; g, nu_hat = ν̂), T; populations = [:E, :I], vars = [:spike])
        mean(order_parameter(x[Population = At(population), Var = At(:spike)]))
    end
end

# begin
#     running.run_parallel(model, params, 2; monitors = ["E.spike"], T, jit = true)
# end

if false # * Plot bifurcation diagram
    f = Figure()
    ax = Axis(f[1, 1])
    heatmap!(ax, X)
    f
end
