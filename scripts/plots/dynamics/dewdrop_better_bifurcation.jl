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
    model = Dewdrop.models.Dewdrop
    begin # FNS parameters
        dx = 1.0
        rho = 20000.0
        kernel = Dewdrop.distances.ExponentialKernel
        J_e = 0.0008
        # delta = 4.0
        # nu = 7.0
        n_ext = 100

        sigma_ee = 0.070
        sigma_ei = 0.080
        sigma_ie = 0.15
        sigma_ii = 0.15

        K_ee = 136
        K_ie = 144
        K_ei = 200
        K_ii = 224
    end
end
begin
    tmax = 25u"s" # * Bump up
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (; dx, rho, kernel, J_e, n_ext,
                    sigma_ee, sigma_ei, sigma_ie, sigma_ii,
                    K_ee, K_ie, K_ei, K_ii)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz
    mua_func = Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt)))

    dn = floor(Int, sqrt(rho / dx^2))
    positions = Dewdrop.positions.GridPositions((dx, dx))((dn, dn))  # Maybe think about capturing this somehow
    positions = map(positions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end
    radius = 0.1 # mm
    origin = [dx / 2, dx / 2]
    mask = map(positions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    local_idxs = findall(mask) |> Dewdrop.numpy.asarray

    monitors = [("E.spike", local_idxs), ("E.input", local_idxs)] .|> pytuple |> pytuple
    stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
                      "susceptibility" => Dewdrop.stats.susceptibility(bin = 10),
                      # "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => Dewdrop.stats.temporal_average,
                      "grand_distribution" => Dewdrop.stats.grand_distribution(n_bins = 1000),
                      "mua" => mua_func)

    metadata = (; positions, local_idxs, mua_dt, tmax, tmin, monitors)
end
begin# * Generate dict of parameter vectors
    sweep = (;
             delta = range(2.0, 8.0, length = 20),
             nu = range(6.5, 6.5, length = 1))
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
    stats, sweep_parameters = Dewdrop.stats.progress_vmap(stats_run, batch_size = 2)(pydict(sweep_params))
end
begin
    Dewdrop.stats.save("dewdrop_better_bifurcation.pickle",
                       (stats, sweep_params, fixed_params, metadata))
end
if false
    begin # * Load stats
        load_stats, sweep_parameters, fixed_parameters, metadata = Dewdrop.stats.load("dewdrop_better_bifurcation.pickle")
    end
    begin
        begin # * Extract a ToolsArray of one statistic
            deltas = sweep_parameters["delta"] |> convert2(Array)
            nus = sweep_parameters["nu"] |> convert2(Array)
            deltas = reshape(deltas, (length(unique(deltas)), length(unique(nus))))
            nus = reshape(nus, (length(unique(deltas)), length(unique(nus))))
            rate = load_stats["rate"]["E.spike"] |> convert2(Array)
            rate = reshape(rate, size(deltas)..., size(rate, 2))
            meanrate = dropdims(mean(rate, dims = 3), dims = 3)
        end
        begin
            f = Figure()
            ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "ν")
            colorrange = extrema(meanrate)
            heatmap!(ax, sort(unique(deltas)), sort(unique(nus)), meanrate; colorrange,
                     colormap = :viridis)
            Colorbar(f[1, 2]; colormap = :viridis, colorrange,
                     label = "mean rate (Hz)")
            f
        end
    end

    begin # * Load mua
        _mua = stats["mua"]["E.spike"] |> convert2(Array)
        ts = range(tmin, stop = tmax, step = uconvert(u"s", mua_dt))[2:end] |> ustripall
        deltas = sweep_parameters["delta"] |> convert2(Array)
    end
    begin
        mua = Timeseries(ts, Dim{:delta}(deltas), _mua')
        mua = mua .- mean(mua, dims = 𝑡)
        spectra = spectrum(mua)
    end
end
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
