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
using Random
using SparseArrays
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin # * Sampling parameters
    batch_size = 1
    path = datadir("critical_sweep")
end

begin # * Fixed parameters
    model = Dewdrop.models.Spatial
    default_params = Dewdrop.defaults(model)

    tmax = 35u"s"
    transient = 5u"s" # The transient. Simulations always begin at 0
    dx = default_params[:parameters][:dx]
    rho = default_params[:parameters][:rho]

    monitors = ["E.spike", "E.input"] |> pytuple
    stat_funcs = Dict("monitor" => Dewdrop.stats.monitor)

    dt = pyconvert(Float32, Dewdrop.brainpy.share["dt"]) * u"ms"
    metadata = (; tmax, transient, monitors, dt)
end

begin # And format parameters
    delta = Dim{:delta}(range(2, 5, length = 32))
    obs = Obs(1:1)
end
begin
    exists = parse_savename.(readdir(path); connector = string(connector))
    exists = getindex.(exists, 2)
    exists = getindex.(exists, "delta")

    delta = delta[round.(delta; sigdigits = 3) .∉ Ref(exists)]
    parameter_grid = map(first, Iterators.product(delta, obs))
    batches = Iterators.partition(parameter_grid[:], batch_size)
end
begin end
begin# * Run simulation
    for (i, delta) in enumerate(batches)
        @info "Batch $i / $(length(batches))"
        keys = rand(UInt32, length(delta))
        jax_keys = Dewdrop.jax.numpy.stack([Dewdrop.PRNGKey(k) for k in keys]) # ? Important, must be python array
        params = (; delta, key = jax_keys)

        runner = Dewdrop.create_run(model; monitors, tmax, transient)
        stats_run = Dewdrop.create_stats_run(runner, stat_funcs)
        stats, sweep_parameters = Dewdrop.partial_vmap(stats_run)(params)
        res = Dewdrop.batchformat(pyconvert(Dict, stats), sweep_parameters; metadata)

        for i in eachindex(delta)
            d = params[:delta][i]
            ps = (; delta = d, key = keys[i])
            merged_params = (; default_params[:parameters]..., ps...)

            r = res.monitor[(; delta = d)]

            begin # * Input statistics
                x = r.var"E.input"
                x = x .- mean(x, dims = 𝑡) # Remove DC offset
                # * MAD of inputs
                τs = logrange(10, 1u"s" / uconvert(u"s", dt), length = 100)
                τs = round.(Int, τs) |> unique
                mad = map(eachslice(x, dims = Neuron)) do _x
                    madev(_x, τs)
                end |> stack

                # * PSD of inputs
                psd = map(eachslice(x, dims = Neuron)) do _x
                    _x = _x .- mean(_x)
                    spectrum(_x, 0.25)
                end |> stack

                # * Downsample for distribution estimate
                distribution = x[1:10:end, :]
            end

            out = Dict("parameters" => merged_params,
                       "spikes" => r.var"E.spike",
                       "inputs/mad" => mad,
                       "inputs/psd" => psd,
                       "inputs/distribution" => distribution)
            filename = savename(ps, "jld2"; connector)
            tagsave(joinpath(path, filename), out)
            r = []
            x = []
            out = []
            GC.gc()
        end

        stats = []
        res = []
        out = []
        runner = []
        stats_run = []
        GC.gc()
    end
end
