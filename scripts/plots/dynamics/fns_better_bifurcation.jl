#! /bin/bash

# -*- mode: julia -*-
#=
exec $HOME/build/julia-1.11.2/bin/julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# $HOME/build/julia-1.11.2/bin/julia maybe
using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin
    model = Dewdrop.models.FNS
    begin # FNS parameters
        N_e = 4000
        J_e = 0.0008 # Microsiemens
        n_ext = 100
        omega_ee = 0.036 * 4000 / N_e
        omega_ie = 0.0564 * 4000 / N_e
        omega_ei = 0.0444 * 4000 / N_e
        omega_ii = 0.078 * 4000 / N_e
    end
end
begin
    tmax = 10u"s"
    tmin = 1u"s" # The transient. Simulations always begin at 0
    fixed_params = (; N_e, J_e, n_ext, omega_ee, omega_ie, omega_ei, omega_ii)

    monitors = ("E.spike", "E.V", "E.input")
    stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
                      "susceptibility" => Dewdrop.stats.susceptibility(10),
                      "spike_spectrum" => Dewdrop.stats.spike_spectrum(10),
                      "temporal_average" => Dewdrop.stats.temporal_average,
                      "grand_distribution" => Dewdrop.stats.grand_distribution(1000))
end
begin# * Generate dict of parameter vectors
    sweep = (; delta = range(1.0, 6.0, length = 10),
             nu = range(1.0, 10.0, length = 10))
    pnames = map(string, keys(sweep))
    pvals = stack(Iterators.product(values(sweep)...), dims = 1)
    sweep_params = Dict(zip(pnames, eachcol(pvals))) # Now a good shape for jax
end
begin # * Create sweep function
    run = Dewdrop.stats.create_run(model, pydict(fixed_params), monitors,
                                   ustrip(to_ms(tmax)),
                                   ustrip(to_ms(tmin)))
    stats_run = Dewdrop.stats.create_stats_run(run, pydict(stat_funcs))
end
begin # * Run simulation
    stats, sweep_parameters = Dewdrop.stats.progress_vmap(stats_run, batch_size = 1)(pydict(sweep_params))
end
begin
    # save("fns_better_bifurcation.jld2", (@strdict stats)) # * Need to convert from python,
    # recursively
    Dewdrop.stats.save("fns_better_bifurcation.pickle",
                       (stats, sweep_parameters, fixed_parameters))
end
begin # * Load stats
    load_stats, sweep_parameters, fixed_parameters = Dewdrop.stats.load("fns_better_bifurcation.pickle")
end
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
