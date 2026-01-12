#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using WorkingRegime
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
WorkingRegime.@preamble
set_theme!(foresight(:physics))

begin # * Load sweep parameters
    files = readdir(datadir("critical_sweep"), join = true)
    ps = map(f -> parse_savename(f; connector = string(connector))[2], files)
    deltas = [p["delta"] for p in ps]
end

begin # * Load PSD data
    psd = map(Chart(Threaded(), ProgressLogger()), files) do f
        load(f, "inputs/psd")
    end
    psd = ToolsArray(psd, Dim{:δ}(deltas)) |> stack
end

begin # * Plot Psd for one delta
    p = median(psd[δ = Near(4.0)], dims = Neuron)
    p = dropdims(p, dims = Neuron)

    f = Figure()
    ax = Axis(f[1, 1]; xscale = log10, yscale = log10)
    lines!(ax, ustripall(p)[𝑓 = eps() .. 150])
    display(f)
end

function susceptibility(x; dt = 100u"ms")
    # * First bin into ms bins
    X = groupby(x, 𝑡 => Base.Fix2(WorkingRegime.group_dt, dt))

    # * Active neurons
    X = map(X) do x
        sum(x, dims = 𝑡) .> 0
    end

    # * Fraction of active neurons at each time step
    rho = mean.(X)

    # * Susceptibility
    chi = mean(rho .^ 2) - mean(rho)^2
end

begin # * Calculate susceptibility
    spikes = map(Chart(Threaded(), ProgressLogger()), files) do f
        load(f, "spikes")
    end
    χ = map(susceptibility, Chart(Threaded(), ProgressLogger()), spikes)
    χ = ToolsArray(χ, Dim{:δ}(deltas)) |> stack
end
begin
    lines(χ)
end
