using Statistics
using TimeseriesTools
using DimensionalData
using IntervalSets
using Unitful
export bprun, bpsolve, bpsweep, Neuron, Population

DimensionalData.@dim Neuron ToolsDim
DimensionalData.@dim Population ToolsDim

function bprun(net::Py, time; monitors = ("E.spike", "I.spike", "E.V", "I.V"), jit = true,
               kwargs...)
    runner = brainpy.DSRunner(net; monitors, jit, kwargs...)
    runner.run(time)
    return runner
end
function bpsolve(net::Py, time::Quantity; kwargs...)
    bpsolve(net, ustrip(uconvert(u"ms", time)); kwargs...)
end

"""
Extract the monitored variables as julia arrays
"""
function bpformat(runner; populations, vars, transient)
    t = runner.mon["ts"].view() |> convert2(Vector)
    dt = brainpy.share["dt"] |> convert2(Float64)
    lastt = last(t) |> convert2(Float64)
    @assert dt ≈ mean(diff(t))
    t = range(start = first(t), step = dt, length = length(t))
    @assert last(t) ≈ lastt
    t = t .* u"ms" # Add time units
    t = t .+ step(t) |> 𝑡 # Correct for python starting at 0
    ps = Iterators.product(populations, vars)
    monitors = popvars2monitors(populations, vars)
    _X = [runner.mon[m].view() |> convert2(PyArray) for m in monitors]
    X = map(_X, ps) do x, (p, v)
        vs = Symbol.(["$p"] .* string.(1:size(x, 2)))
        Timeseries(t, Neuron(vs), x)[𝑡(OpenInterval(transient, Inf * u"s"))]
    end
    X = reshape(X, length(populations), length(vars))
    X = ToolsArray(X, (Population(populations), Var(vars)))
end

function popvars2monitors(populations, vars)
    ps = collect(Iterators.product(populations, vars))[:]
    monitors = ["$p.$v" for (p, v) in ps] |> Tuple
    return monitors
end
function monitors2popvars(monitors)
    ps = split.(monitors, ".")
    ps = [(Symbol(p[1]), Symbol(p[2])) for p in ps]
    populations = first.(ps) |> unique |> collect
    vars = last.(ps) |> unique |> collect
    return return populations, vars
end
function bpsolve(net::Py, time::Real; populations = [:E, :I], vars = [:V],
                 transient = 500u"ms",
                 inputs = nothing,
                 kwargs...)
    monitors = popvars2monitors(populations, vars)
    if !isnothing(inputs)
        inputs = models.format_input(inputs)
    end
    runner = bprun(net, time; monitors, inputs, kwargs...)
    return bpformat(runner; populations, vars, transient)
end

function bpformat(res, param::Symbol, vals; dt, transient, monitors)
    # ? res has first dim == 'monitors'. Each element has shape (delta, ts, neurons)
    populations, vars = monitors2popvars(monitors)
    nt = res[0][0].shape[0] |> convert2(Int)
    t = range(start = dt, step = dt, length = nt) .* u"ms"

    _X = [m |> convert2(PyArray) for m in res]
    X = map(_X, monitors) do x, m
        p = first(split(m, "."))
        vs = Symbol.(["$p"] .* string.(1:size(x)[end]))
        x = ToolsArray(x, (Dim{param}(vals), 𝑡(t), Neuron(vs)))
        return x[𝑡(OpenInterval(transient,
                                Inf * u"s"))]
    end
    X = map(Iterators.product(populations, vars)) do (p, v) # Get the monitors in the right order
        return X[monitors .== ["$p.$v"]] |> only
    end
    X = reshape(X, length(populations), length(vars))
    return ToolsArray(X, (Population(populations), Var(vars))) |> stack
end
function _bpsweep(model, param::Val{:delta}, vals;
                  duration,
                  transient,
                  populations = [:E, :I],
                  vars = [:V],
                  num_parallel = 10)
    monitors = popvars2monitors(populations, vars)
    duration = uconvert(u"ms", duration) |> ustrip
    dt = brainpy.share["dt"] |> convert2(Float64)
    res = model.sweep_deltas(jax.numpy.array(vals);
                             duration,
                             monitors,
                             num_parallel)
    param = typeof(param).parameters |> only
    X = bpformat(res, param, vals; dt, transient, monitors)
    return X
end
function bpsweep(model, param::Symbol, vals; kwargs...)
    _bpsweep(model, Val(param), vals; kwargs...)
end
function bpsweep(model, param::Pair; kwargs...)
    _bpsweep(model, Val(first(param)), last(param); kwargs...)
end

function bpsweep(model_class, conn::Py, params::Dict, param::Pair; batch_size,
                 batch_seed = 42,
                 kwargs...)
    vals = last(param)
    param = first(param)
    # * Subset the vals into batches of size batch
    batch_vals = Iterators.partition(vals, batch_size)
    X = map(batch_vals) do bvals
        clear_live_arrays()
        model = model_class(; params..., copy_conn = conn,
                            key = jax.random.PRNGKey(batch_seed)) # Fast construction
        _bpsweep(model, Val(param), bvals; kwargs...)
    end
    pcat(x, y) = cat(x, y, dims = param)
    return reduce(pcat, X)
end

""" Only for jittable funcs. Each value fo params should be an interator of the same length"""
function bpsweep(func, params::Dict; batch_size = 5, kwargs...)
    return brainpy.running.jax_vectorize_map(func, params, num_parallel = batch_size)
end
