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
    computed_hashes = readdir(datadir("spatial_sampling")) .|> splitext .|> first
    computed_hashes = filter(x -> all(isdigit, x), computed_hashes)
    computed_hashes = map(Base.Fix1(parse, UInt), computed_hashes)
    Q = map(x -> false, hash_grid)
    for h in computed_hashes
        if h in hash_grid
            Q[findfirst(==(h), hash_grid)] = true
        end
    end
end
begin # * Loop through files and plot
    @info "Processing $(sum(Q)) files"
    Q = map(Chart(ProgressLogger()), hash_grid[Q]) do hash
        file = string(hash) * ".jld2"

        if isfile(datadir("spatial_sampling", file))
            parameters = load(datadir("spatial_sampling", file), "parameters")
            title = parameters[[:delta, :K_ee, :K_ei, :K_ie, :K_ii]]
            title = map(keys(title), values(title)) do k, v
                        k => round(v, sigdigits = 3)
                    end |> NamedTuple |> Dewdrop.sortparams |> string

            animation_file = plotdir("spatial_sampling", "animation", "$hash.mp4")
            if isfile(animation_file)
                @info "Animation for $hash already exists"
            else
                spikes = load(datadir("spatial_sampling", file), "monitor")["E.spike"]
                rates = Dewdrop.compute_rates(spikes, 50u"ms")
                Dewdrop.animate_rates(rates, dx;
                                      filename = animation_file,
                                      axis = (; title, titlesize = 10))
            end

            # * Now do MUA spectrum
            mua_file = plotdir("spatial_sampling", "mua_spectrum", "$hash.png")
            if isfile(mua_file)
                @info "MUA spectrum for $hash already exists"
            else
                mua = load(datadir("spatial_sampling", file), "mua")["E.spike"]
                s = spectrum(mua, 0.5u"Hz")
                f = plotspectrum(s)
                wsave(mua_file, f)
            end

            return true
        else
            @warn "File $file not found but quality check indicated it should be present"
            return false
        end
    end
end
