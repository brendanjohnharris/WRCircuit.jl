#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
Dewdrop.@preamble
set_theme!(foresight(:physics))

model = models.FNScircuit
modelname = "FNScircuit"
T = 5500.0
zetas = range(1, 40, length = 2)
m = model(; rho = 30000, nu = 100.0, n_ext = 30, J_e = 0.0005, zeta = first(zetas),
          key = jax.random.PRNGKey(rand(UInt32)))
N = m.E.size |> convert2(Vector)
domain = m.E.embedding.domain |> convert2(Vector)
dx = domain ./ N
xs = range.(0 .+ dx / 2, domain .- dx / 2, N)
out = map(zetas) do zeta
    m.reinit_weights(zeta)
    res = bpsolve(m, T; populations = [:E], vars = [:spike])
    spikes = res[Var = At(:spike)][Population = At(:E)]
    begin # * susecptibility
        x = spikes[1, :]

        dt = 10u"ms"
        x = sum.(coarsegrain(spikes, dt)) # Bin over time
        x = map(eachslice(x, dims = 1)) do x
            ToolsArray(reshape(x, N...), (X(xs[1]), Y(xs[2])))
        end |> stack
        x = permutedims(x, (3, 1, 2))
        x = set(x, 𝑡 => mean.(times(x)))
        ρ = mean(x .> 0, dims = (1, 2)) # Fraction of active neurons
        χ = mean(ρ .^ 2) - mean(ρ)^2
    end
    return χ
end
out = ToolsArray(out, (Dim{:zeta}(zetas),))
save("fns_bifurcation.jld2", Dict("out" => out))

# begin
#     file = "fns_bifurcation.jld2"
#     f = jldopen(file, "r")
# end
