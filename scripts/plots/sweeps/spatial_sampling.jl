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
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin # * Sampling parameters
    batch_size = 4
    path = datadir("spatial_sampling")
end

begin # * Fixed parameters
    model = Dewdrop.models.Spatial
    default_params = Dewdrop.defaults(model)

    tmax = 50u"s" # * Bump up
    transient = 10u"s" # The transient. Simulations always begin at 0
    dx = 0.5
    rho = 20000.0
    key = 42

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz
    mua_func = Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt)))

    dn = round(Int, sqrt(rho * dx^2))
    positions = Dewdrop.positions.GridPositions((dx, dx))((dn, dn))  # Maybe think about capturing this somehow
    positions = map(positions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end

    # * Spatial binning
    nbins = 20
    edges = range(0, dx, nbins + 1)
    ix = [clamp(searchsortedlast(edges, pos[1]), 1, nbins) for pos in positions]
    iy = [clamp(searchsortedlast(edges, pos[2]), 1, nbins) for pos in positions]
    bin_indices = [Int[] for _ in 1:nbins, _ in 1:nbins]
    for (neuron_idx, (bx, by)) in enumerate(zip(ix, iy))
        push!(bin_indices[bx, by], neuron_idx)
    end

    radius = 0.1 # mm
    origin = [dx / 2, dx / 2]
    mask = map(positions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    local_idxs = findall(mask) |> Dewdrop.numpy.asarray

    monitors = ["E.spike"] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10),
    #                   #   "radial_autocorrelation" => Dewdrop.stats.radial_autocorrelation(positions,
    #                   #                                                                    0.05))#,
    #                   #  "efficiency" => Dewdrop.stats.efficiency(bin_indices, 1000))
    #                   # "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
    #                   #  "temporal_average" => Dewdrop.stats.temporal_average,
    #                   #   "grand_distribution" => Dewdrop.stats.grand_distribution(n_bins = 1000),
    #                   "mua" => mua_func)
    stat_funcs = Dict(#"rate" => Dewdrop.stats.firing_rate,
                      #   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10),
                      #   "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => Dewdrop.stats.temporal_average,
                      #   "mua" => Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt))),
                      "monitor" => Dewdrop.stats.monitor)

    metadata = (; positions, bin_indices, mua_dt, tmax, transient, monitors)
end

begin # * The idea is to randomly sample some parameters from discretized parameter distributions. This script can be left running to populate the model cache.
    # * Start by defining parameter ranges to sample from
    parameter_grid = [
        Dim{:delta}(range(1, 9, length = 32)),
        Dim{:nu}(range(2, 6, length = 16))
    ]

    parameter_grid = map(parameter_grid) do d
        round.(d, sigdigits = 3)
    end

    parameter_grid = Iterators.product(parameter_grid...) |> collect
    parameter_grid = map(parameter_grid) do ps
        NamedTuple(name.(dims(parameter_grid)) .=> ps)
    end

    hash_grid = map(parameter_grid) do p
        merged_params = (; default_params[:parameters]..., p, key = [0, key])
        hash(params)
    end

    tagsave(joinpath(path, "parameter_grid.jld2"),
            Dict("parameter_grid" => parameter_grid, "default_params" => default_params,
                 "metadata" => metadata, "hash_grid" => hash_grid))
end

while true
    begin # * Load computed grid
        computed_hashes = splitext.(readdir(path)) .|> first .|> Meta.parse
        Q = map(hash_grid) do h
            h in computed_hashes
        end
    end
    if all(Q)
        @info "All parameter combinations computed, exiting"
        break
    end
    begin # * Then, randomly select batches of un-computed parameter combinations
        these_ps = shuffle(parameter_grid[.!Q])[1:min(batch_size, count(!, Q))]
    end
    begin # And format parameters
        jax_keys = Dewdrop.jax.numpy.stack([Dewdrop.PRNGKey(key) for _ in 1:batch_size]) # Important, must be python array

        pnames = keys(these_ps |> first)
        these_ps = map(pnames) do n
            n => getindex.(these_ps, n)
        end |> Dict{Symbol, Any}
        push!(these_ps, :key => jax_keys)
    end

    begin # * And run the batch simulation
        begin # * Create sweep function
            run = Dewdrop.create_run(model; monitors, tmax, transient)
            stats_run = Dewdrop.create_stats_run(run, stat_funcs)
        end
        begin # * Run simulation
            stats, sweep_parameters = Dewdrop.partial_vmap(stats_run)(these_ps)
        end
    end
    begin # * Format the monitors
        # watchkeys = pyconvert(Vector{String}, stats.keys())
        res = Dewdrop.bpformat(stats["monitor"], sweep_parameters; transient, tmax)
    end

    begin # * Save each result, combining swept params with default model values
        map(collect(res)) do (params, r)
            merged_params = (; default_params[:parameters]..., params..., key = [0, key])
            filename = hash(merged_params)
            out = Dict("parameters" => merged_params, r...)
            tagsave(joinpath(path, "$filename.jld2"), out)
            return merged_params
        end
    end
end
