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
        kernel = Dewdrop.distances.GaussianKernel
        J_e = 0.0007
        delta = 3.0
        nu = 8.0
        n_ext = 100

        sigma_ee = 0.04
        sigma_ei = 0.05
        sigma_ie = 0.12
        sigma_ii = 0.12

        K_ee = 72
        K_ie = 66
        K_ei = 96
        K_ii = 126
    end
end
begin
    tmax = 15u"s" # * Bump up
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (; dx, rho, kernel, J_e, n_ext,
                    sigma_ee, sigma_ei, sigma_ie, sigma_ii,
                    K_ee, K_ie, K_ei, K_ii)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz
    mua_func = Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt)))

    dn = round(Int, sqrt(rho * dx^2))
    positions = Dewdrop.positions.GridPositions((dx, dx))((dn, dn))  # Maybe think about capturing this somehow
    positions = map(positions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end

    # * Spatial binnin
    nbins = 20
    edges = range(0, dx, nbins + 1)
    ix = [clamp(searchsortedlast(edges, pos[1]), 1, nbins) for pos in positions]
    iy = [clamp(searchsortedlast(edges, pos[2]), 1, nbins) for pos in positions]
    bin_indices = [Int[] for _ in 1:nbins, _ in 1:nbins]
    for (neuron_idx, (bx, by)) in enumerate(zip(ix, iy))
        push!(bin_indices[bx, by], neuron_idx)
    end

    # radius = 0.1 # mm
    # origin = [dx / 2, dx / 2]
    # mask = map(positions) do pos
    #     dp = abs.(pos .- origin)
    #     dp = min.(dp, dx .- dp)
    #     norm(dp) < radius
    # end # scatter(positions.|> Point2f, color=mask) to check
    # local_idxs = findall(mask) |> Dewdrop.numpy.asarray

    monitors = ["E.spike"] |> pytuple
    stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
                      "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
    #   "radial_autocorrelation" => Dewdrop.stats.radial_autocorrelation(positions,
    #                                                                    0.05))#,
    #   "efficiency" => Dewdrop.stats.efficiency(bin_indices, 1000))
    # "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
    #   "temporal_average" => Dewdrop.stats.temporal_average,
    #   "grand_distribution" => Dewdrop.stats.grand_distribution(n_bins = 1000),
    #   "mua" => mua_func)

    metadata = (; positions, bin_indices, mua_dt, tmax, tmin, monitors)
end
begin# * Generate dict of parameter vectors
    sweep = (;
             delta = range(2.0, 8.0, length = 12),
             nu = range(9.5, 9.5, length = 1))
    pnames = map(string, keys(sweep))
    pvals = stack(Iterators.product(values(sweep)...), dims = 1)
    sweep_params = Dict{String, Any}(zip(pnames, eachcol(pvals))) # Now a good shape for jax
    n_iters = length(first(values(sweep_params)))
    jax_keys = Dewdrop.jax.random.split(Dewdrop.jax.random.PRNGKey(42), n_iters)
    sweep_params["key"] = Dewdrop.numpy.array.(jax_keys) # * So that each run is independent
end
begin # * Create sweep function
    run = Dewdrop.stats.create_run(model, pydict(fixed_params), monitors,
                                   ustrip(to_ms(tmax)),
                                   ustrip(to_ms(tmin)))
    stats_run = Dewdrop.stats.create_stats_run(run, pydict(stat_funcs))
end
begin # * Run simulation
    stats, sweep_parameters = Dewdrop.stats.progress_vmap(stats_run, batch_size = 6)(pydict(sweep_params))
end
begin
    Dewdrop.stats.save("spatial_sweep.pickle",
                       (stats, sweep_params, fixed_params, metadata))
end
# if false
#     begin # * Load stats
#         load_stats, sweep_parameters, fixed_parameters, metadata = Dewdrop.stats.load("dewdrop_better_bifurcation.pickle")
#     end
#     begin
#         begin # * Extract a ToolsArray of one statistic
#             deltas = sweep_parameters["delta"] |> convert2(Array)
#             nus = sweep_parameters["nu"] |> convert2(Array)
#             deltas = reshape(deltas, (length(unique(deltas)), length(unique(nus))))
#             nus = reshape(nus, (length(unique(deltas)), length(unique(nus))))
#             rate = load_stats["rate"]["E.spike"] |> convert2(Array)
#             rate = reshape(rate, size(deltas)..., size(rate, 2))
#             meanrate = dropdims(mean(rate, dims = 3), dims = 3)
#         end
#         begin
#             f = Figure()
#             ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "ν")
#             colorrange = extrema(meanrate)
#             heatmap!(ax, sort(unique(deltas)), sort(unique(nus)), meanrate; colorrange,
#                      colormap = :viridis)
#             Colorbar(f[1, 2]; colormap = :viridis, colorrange,
#                      label = "mean rate (Hz)")
#             f
#         end
#     end

#     begin # * Load mua
#         _mua = stats["mua"]["E.spike"] |> convert2(Array)
#         ts = range(tmin, stop = tmax, step = uconvert(u"s", mua_dt))[2:end] |> ustripall
#         deltas = sweep_parameters["delta"] |> convert2(Array)
#     end
#     begin
#         mua = Timeseries(ts, Dim{:delta}(deltas), _mua')
#         mua = mua .- mean(mua, dims = 𝑡)
#         spectra = spectrum(mua)
#     end
# end
# begin
#     f = Figure(size = (800, 600))
#     colorrange = extrema(lookup(stats, :omicron)) .+ [0, 0.001]
#     map(enumerate(eachslice(ustripall(stats), dims = :statistic))) do (i, stat)
#         statname = refdims(stat) |> only |> only
#         ax = Axis(f[i, 1]; ylabel = "$statname")
#         map(eachslice(stat, dims = :omicron)) do stat
#             omicron = refdims(stat)
#             omicron = omicron[dimname.(omicron) .== ["omicron"]] |> only |> only
#             stat = upsample(stat, 10; dims = 1)
#             lines!(ax, stat; color = omicron, colorrange, label = "o = $omicron")
#         end
#         axislegend(ax)
#     end
#     map(contents(f.layout)[1:(end - 1)]) do ax
#         hidexdecorations!(ax)
#         ax.xgridvisible = true
#         ax.xticksvisible = true
#     end
#     last(contents(f.layout)).xlabel = "δ"
#     save("fns_bifurcation.pdf", f)
#     f
# end
