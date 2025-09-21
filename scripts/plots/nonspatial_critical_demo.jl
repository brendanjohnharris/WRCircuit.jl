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
    model = Dewdrop.models.Nonspatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        sigma_ee = 0.075
        sigma_ei = 0.095
        sigma_ie = 0.19
        sigma_ii = 0.19
        k = 300 / 270
        K_ee = 390 # 270
        K_ei = round(Int, k * 420)
        K_ie = round(Int, k * 180)
        K_ii = round(Int, k * 200)
        delta = 3.0
        nu = 3.65
        n_ext = 100
        J_ee = 0.001 # 0.00105
        J_ei = 0.00145
        tau_r_i = 2.0
        tau_d_i = 5.0
    end
end

begin
    tmax = 10u"s" # * Bump up
    tmin = 5u"s" # The transient. Simulations always begin at 0
    fixed_params = (; rho,
                    dx,
                    sigma_ee,
                    sigma_ei,
                    sigma_ie,
                    sigma_ii,
                    K_ee,
                    K_ei,
                    K_ie,
                    K_ii,
                    delta,
                    nu,
                    n_ext,
                    J_ee, # 0.00105
                    J_ei,
                    tau_r_i,
                    tau_d_i)

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, tmax; populations = [:E, :I], vars = [:spike, :V],
                transient = tmin)
end


begin # * Membrane potential
    V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
    V = set(V, 𝑡 => uconvert.(u"s", times(V)))
    V = rectify(V, dims = 𝑡)
    lines(V[1:2000, 970]) |> display
end
begin # * Unit spectrum
    s = spectrum(V)
    s = mean(s, dims = Neuron)
    s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 2 .. 200]
    lines(s; axis = (; xscale = log10, yscale = log10)) |> display
end
begin # * MUA spectrum
    mdt = 3.0u"ms"
    mua = groupby(spikes, 𝑡 => Base.Fix2(group_dt, mdt))
    mua = map(mua) do r
        dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", dt)
    end |> stack
    mua = permutedims(mua, (𝑡, Neuron))
    mua = rectify(mua, dims = 𝑡)
    N = lookup(mua, Neuron) |> length |> sqrt |> Int
    ts = dims(mua, 𝑡)
    mua = reshape(mua, (size(mua, 1), N, N))
    idxs = [i:(i + 9) for i in 1:10:(N - 10 + 1)]
    idxs = Iterators.product(idxs, idxs)
    muas = map(idxs) do (i, j)
        m = mua[:, i, j] # * Get local patch
        m = mean(m, dims = (2, 3))
        m = ToolsArray(vec(m), ts)
        m = set(m, 𝑡 => uconvert.(u"s", times(m)))
        m = rectify(m, dims = 𝑡)
        return spectrum(m)
    end
    muas = mean(muas)
    lines(ustripall(muas)[𝑓 = 1 .. 200]; axis = (; xscale = log10, yscale = log10)) |>
    display
end
begin # * Plot inter-spike interval distribution
    spiketimes = map(eachslice(spikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end
    isis = map(spiketimes) do sts
        diff(sts)
    end
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Inter-spike interval (ms)",
              ylabel = "Density", limits = ((0, 100), nothing))
    hist!(ax, Iterators.flatten(isis) |> collect |> ustrip, normalization = :pdf,
          bins = range(0.0, 1000, step = 5))
    display(f)
end
