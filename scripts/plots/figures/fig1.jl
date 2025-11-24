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
using Optim
using MoreMaps
Dewdrop.@preamble
set_theme!(foresight(:physics))

# begin # * Plot inputs heatmap
#     file = files[1]
#     input = load(file, "inputs/distribution")
#     X = input[𝑡 = 20u"s" .. 20.1u"s"]
#     X = mean(X, dims = 𝑡)
#     X = dropdims(X, dims = 𝑡)
#     dx = load(file, "parameters")[:dx]
#     positions, idxs = Dewdrop.infer_geometry(X, dx)
#     N = sqrt(length(X)) |> Int
#     X = reshape(X, N, N)
#     heatmap(X;)
# end
begin # * Run a sample simulation at the working point (default values)
    tmax = 30u"s" # * Bump up
    tmin = 5u"s"
    model = Dewdrop.models.Spatial

    m = model(;)
    x = bpsolve(m, tmax;
                populations = [:E],
                vars = [:spike, :V, :input],
                transient = tmin)
end

begin # * Animate
    spikes = x[Population = At(:E), Var = At(:spike)]
    rates = Dewdrop.compute_rates(spikes, 50u"ms")
    dx = Dewdrop.defaults(model)[:parameters][:dx]
    Dewdrop.animate_rates(rates, dx; filename = plotdir("fig1", "animation.mp4"))
end

