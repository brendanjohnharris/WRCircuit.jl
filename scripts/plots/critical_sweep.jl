#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.12 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# Critical sweep — exponent extraction.
#
# Load every (δ, Δg_K) simulation in data/critical_sweep, fit the per-neuron
# diffusion exponent (from the input MAD) and spectral exponent (from the input
# PSD), and save *all* per-neuron exponents (not neuron-averages) to
# data/plots/critical_sweep.jld2.
#
# The fitting recipe follows scripts/plots/_critical_sweep.jl: the diffusion
# exponent is the first component of a 2-component MAPPLE fit to the MAD curve;
# the spectral exponent is the last component of a 1-component fit to the
# 10-1000 Hz PSD.

using DrWatson
DrWatson.@quickactivate
using WRCircuit
using JLD2
using LinearAlgebra
using Optim
using MoreMaps
using Statistics
WRCircuit.@preamble
set_theme!(foresight(:physics))

# Neurons to fit per cell. 1 = every neuron ("all exponents"); the prototype
# used 50 purely for speed. Bump this if the full fit is too slow.
const neuron_step = 50

# ──────────────────────────────────────────────────────────────────────────────
# Per-neuron exponent fits
# ──────────────────────────────────────────────────────────────────────────────

"Diffusion exponents (one per neuron): first component of a 2-component MAPPLE fit to MAD."
function diffusion_exponents(mad; step = neuron_step)
    return map(eachcol(mad[:, 1:step:end])) do y
        try
            y = ustripall(y)
            m = fit(MAPPLE, y; components = 2, peaks = 0)
            fit!(m, y)
            return first(m.params.components.β)
        catch
            return NaN
        end
    end
end

"Spectral exponents (one per neuron): last component of a 1-component MAPPLE fit to the 10-1000 Hz PSD."
function spectral_exponents(psd; step = neuron_step)
    return map(eachcol(psd[:, 1:step:end])) do y
        try
            _p = y[𝑓 = 10u"Hz" .. 1000u"Hz"]
            _p = logsample(ustripall(_p))
            m = fit(MAPPLE, _p; components = 1, peaks = 0)
            fit!(m, _p)
            return last(m.params.components.β)
        catch
            return NaN
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Assemble the full (δ, Δg_K) grid of files
# ──────────────────────────────────────────────────────────────────────────────

begin # * Load sweep parameters (mirrors _critical_sweep.jl, but the full grid)
    files = readdir(datadir("critical_sweep"), join = true)
    ps = map(files) do f
        fname = parse_savename(f; connector = string(connector))[2]
        if haskey(fname, "key") || !haskey(fname, "delta") || !haskey(fname, "Delta_g_K")
            return nothing
        else
            return fname
        end
    end
    keep = .!isnothing.(ps)
    files = files[keep]
    ps = ps[keep]

    deltas = [p["delta"] for p in ps]
    Delta_g_Ks = [p["Delta_g_K"] for p in ps]

    # Sorted lookups so the saved grid (and any heatmap) is monotonic in both axes
    udelta = sort(unique(deltas))
    ugk = sort(unique(Delta_g_Ks))

    parameter_grid = Iterators.product(
        Dim{:delta}(udelta),
        Dim{:Delta_g_K}(ugk)
    ) |> collect
    parameter_grid = map(parameter_grid) do (d, gk)
        idx = findfirst((deltas .== d) .& (Delta_g_Ks .== gk))
        isnothing(idx) ? nothing : files[idx]
    end
    @info "Found $(count(!isnothing, parameter_grid)) / $(length(parameter_grid)) grid cells"
end

# ──────────────────────────────────────────────────────────────────────────────
# Fit exponents across the grid (one pass; missing cells -> empty vectors)
# ──────────────────────────────────────────────────────────────────────────────

begin # * Diffusion exponents
    a = map(Chart(Threaded(), ProgressLogger()), parameter_grid) do f
        isnothing(f) && return Float64[]
        try
            return diffusion_exponents(load(f, "inputs/mad"))
        catch err
            @warn "Failed MAD fit for $f" err
            return Float64[]
        end
    end
end

begin # * Spectral exponents
    b = map(Chart(Threaded(), ProgressLogger()), parameter_grid) do f
        isnothing(f) && return Float64[]
        try
            return spectral_exponents(load(f, "inputs/psd"))
        catch err
            @warn "Failed PSD fit for $f" err
            return Float64[]
        end
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# Save (per-neuron exponents only, plus the parameter lookups)
# ──────────────────────────────────────────────────────────────────────────────

begin # * Save
    mkpath(datadir("plots"))
    out = Dict(
        "a" => a,            # per-neuron diffusion exponents, ToolsArray{Vector} over (δ, Δg_K)
        "b" => b,            # per-neuron spectral exponents
        "delta" => udelta,
        "Delta_g_K" => ugk
    )
    outfile = datadir("plots", "critical_sweep.jld2")
    tagsave(outfile, out)
    @info "Saved $outfile"
end
