using Statistics
using TimeseriesTools
using DimensionalData
using IntervalSets
using Unitful
export bprun, bpsolve, Neuron, Population

DimensionalData.@dim Neuron ToolsDim
DimensionalData.@dim Population ToolsDim

function bprun(net::Py, time; monitors = ("E.spike", "I.spike", "E.V", "I.V"), jit = true,
               kwargs...)
    runner = brainpy.DSRunner(net; monitors, jit, kwargs...)
    runner.run(time)
    return runner
end

function bpsolve(net::Py, time; populations = [:E, :I], vars = [:V], transient = 500u"ms",
                 kwargs...)
    ps = collect(Iterators.product(populations, vars))[:]
    monitors = ["$p.$v" for (p, v) in ps] |> Tuple
    runner = bprun(net, time; monitors, kwargs...)
    t = runner.mon["ts"].view() |> convert2(Vector)
    dt = brainpy.share["dt"] |> convert2(Float64)
    lastt = last(t) |> convert2(Float64)
    @assert dt ≈ mean(diff(t))
    t = range(start = first(t), step = dt, length = length(t))
    @assert last(t) ≈ lastt
    t = t .* u"ms" # Add time units
    t = t .+ step(t) |> 𝑡 # Correct for python starting at 0
    X = [runner.mon[m].view() |> convert2(Matrix) for m in monitors]
    X = map(X, ps) do x, (p, v)
        vs = Symbol.(["$p"] .* string.(1:size(x, 2)))
        Timeseries(t, Neuron(vs), x)[𝑡(OpenInterval(transient, Inf * u"s"))]
    end
    X = reshape(X, length(populations), length(vars))
    ToolsArray(X, (Population(populations), Var(vars)))
end
