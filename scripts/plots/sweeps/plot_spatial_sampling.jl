#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.11 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
using DataInterpolations
using MoreMaps
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin # * Load manifest
    parameter_grid, hash_grid, default_params = load(datadir("spatial_sampling",
                                                             "parameter_grid.jld2"),
                                                     "parameter_grid", "hash_grid",
                                                     "default_params")
    dx = default_params[:parameters][:dx]
end
begin # * Check file quality
    Q = map(hash_grid) do hash
        file = string(hash) * ".jld2"
        return isfile(datadir("spatial_sampling", file))
    end
end
begin # * Loop through files and plot
    Q = map(Chart(ProgressLogger()), hash_grid) do hash
        file = string(hash) * ".jld2"
        if isfile(datadir("spatial_sampling", file))
            spikes = load(datadir("spatial_sampling", file), "E.spike")
            rates = Dewdrop.compute_rates(spikes, 50u"ms")
            Dewdrop.animate_rates(rates, dx;
                                  filename = plotdir("spatial_sampling", "animation",
                                                     "$hash.mp4"))
            return true
        else
            return false
        end
    end
end
