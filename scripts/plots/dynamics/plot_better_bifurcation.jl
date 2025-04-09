#! /bin/bash
# -*- mode: julia -*-
#=
exec $HOME/build/julia-1.11.2/bin/julia -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
Dewdrop.@preamble
set_theme!(foresight(:physics))

# * Check this with fns_better_bifurcation.jl
tmax = 10u"s"
tmin = 1u"s" # The transient. Simulations always begin at 0
mua_dt = 2u"ms"

begin # * Load stats
    load_stats, sweep_parameters, fixed_parameters = Dewdrop.stats.load("fns_better_bifurcation.pickle")
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
        ax = Axis(f[1, 1]; xlabel = "δ", ylabel = "mean rate (Hz)")
        colorrange = extrema(nus)
        # heatmap!(ax, sort(unique(deltas)), sort(unique(nus)), meanrate; colorrange,
        #  colormap = :viridis)

        traces!(ax, sort(unique(deltas)), sort(unique(nus)), meanrate; colormap = :viridis)

        Colorbar(f[1, 2]; colormap = :viridis, colorrange,
                 label = "background rate (Hz)")
        f
    end
end

begin # * Load mua
    _mua = load_stats["mua"]["E.spike"] |> convert2(Array)
    ts = range(tmin, stop = tmax, step = uconvert(u"s", mua_dt))[2:end] |> ustripall

    delta = sort(unique(deltas))
    nu = sort(unique(nus))

    _mua = reshape(_mua, length(delta), length(nu), size(_mua, 2))
    mua = ToolsArray(_mua, (Dim{:delta}(delta), Dim{:nu}(nu), 𝑡(ts)))
    mua = permutedims(mua, (3, 1, 2))
end
begin
    mumua = mua .- mean(mua, dims = 𝑡)
    spectra = map(x -> spectrum(x, 0.8, padding = 500), eachslice(mumua, dims = (2, 3))) |>
              stack
    plot(spectra[delta = Near(4), nu = Near(7)])
end
begin
    # * plot a heatmap for a given nu
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "frequency (Hz)", ylabel = "δ")
    x = spectra[delta = Near(3.5)]
    # x = x[nu = 1.0 .. 5.0]
    x = x[𝑓 = IntervalSets.OpenInterval(0, 200)]
    x = log10.(x)
    # x = x ./ maximum(x, dims = 2)
    colorrange = extrema(x)
    heatmap!(ax, x, colormap = (pelagic |> reverse))
    f
end

begin # * Load distribution of synaptic weights
    input_distribution, bins = load_stats["grand_distribution"]["E.input"] .|>
                               convert2(Array)

    delta = sort(unique(deltas))
    nu = sort(unique(nus))

    input_distribution = reshape(input_distribution, length(delta), length(nu),
                                 size(input_distribution, 2))
    bins = reshape(bins, length(delta), length(nu), size(bins, 2))

    input_distribution = ToolsArray(input_distribution,
                                    (Dim{:delta}(delta), Dim{:nu}(nu),
                                     Dim{:bin}(1:size(input_distribution, 3))))
    input_distribution = permutedims(input_distribution, (3, 1, 2))
    bins = ToolsArray(bins, (Dim{:delta}(delta), Dim{:nu}(nu), Dim{:bin}(1:size(bins, 3))))
    bins = permutedims(bins, (3, 1, 2))
end
begin
    # * plot a heatmap for a given nu
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Input during timestep", ylabel = "Frequency",
              yscale = log10,
              limits = (nothing, (1, 50000000)))
    x = input_distribution[nu = Near(6)]
    b = bins[nu = Near(6)]

    dd = 5
    map(delta[1:dd:end], eachslice(x, dims = :delta)[1:dd:end],
        eachslice(b, dims = :delta)[1:dd:end]) do _delta, _x, _b
        lines!(ax, collect(_b), (collect(_x)); color = _delta, alpha = 0.7,
               colormap = :turbo, colorrange = extrema(delta))
    end
    Colorbar(f[1, 2]; colormap = :turbo, colorrange = extrema(delta),
             label = "δ")
    f
end
