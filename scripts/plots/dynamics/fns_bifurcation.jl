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

begin
    T = 10000.0
    model = models.FNScircuit
    modelname = "FNScircuit"

    # Shencong Parameters
    delta = 0.007
    dx = 128 * delta # Originally 64*delta # 64 x 64 integer grid, 7um spacing
    rho = 20000 # 12000
    p_ee = 0.8
    p_ei = 0.7
    p_ie = 0.4
    p_ii = 0.57
    sigma_ee = 7.5 * delta
    sigma_ei = 9.5 * delta
    sigma_ie = 19 * delta
    sigma_ii = 19 * delta
    J_e = 0.0008 # Microsiemens. In spontaneous simulation code. Different from main paper
    kernel = models.FNS.ExponentialKernel
    @info "Building model"
    m = model(rho = rho, dx = dx, J_e = J_e,
              nu = 120, n_ext = 25,
              zeta = 4,
              p_ee = p_ee,
              p_ei = p_ei,
              p_ie = p_ie,
              p_ii = p_ii,
              sigma_ee = sigma_ee,
              sigma_ei = sigma_ei,
              sigma_ie = sigma_ie,
              sigma_ii = sigma_ii,
              kernel = kernel,
              key = jax.random.PRNGKey(42))
end

N = m.E.size |> convert2(Vector)
domain = m.E.embedding.domain |> convert2(Vector)
dx = domain ./ N
xs = range.(0 .+ dx / 2, domain .- dx / 2, N)
zetas = range(5, 10, length = 50)
transient = 5000u"ms"
out = map(zetas) do zeta
    @info "ζ = $zeta"
    m.reinit_weights(zeta)
    brainpy.reset_state(m)
    res = bpsolve(m, T; populations = [:E], vars = [:spike], transient)
    spikes = res[Var = At(:spike)][Population = At(:E)]
    begin # * Susceptibility
        dt = 10u"ms"
        x = sum.(coarsegrain(spikes, dt)) # Bin over time
        x = map(eachslice(x, dims = 1)) do x
            ToolsArray(reshape(x, N...), (X(xs[1]), Y(xs[2])))
        end |> stack
        x = permutedims(x, (3, 1, 2))
        x = set(x, 𝑡 => mean.(times(x)))
        ρ = mean(x .> 0, dims = (1, 2)) # Fraction of active neurons
        χ = mean(ρ .^ 2) - mean(ρ)^2
        λ = sum(spikes, dims = 𝑡) ./ duration(spikes)
        λ = uconvert.(u"Hz", mean(λ))
    end
    return χ, λ
end
χ = ToolsArray(first.(out), (Dim{:zeta}(zetas),))
λ = ToolsArray(last.(out), (Dim{:zeta}(zetas),))
save("fns_bifurcation.jld2", (@strdict χ λ))

# begin
#     file = "fns_bifurcation.jld2"
#     f = jldopen(file, "r")
# end
