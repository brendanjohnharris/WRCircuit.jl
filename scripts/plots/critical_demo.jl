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
    model = Dewdrop.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        # sigma_ee = 0.07
        # sigma_ei = 0.095
        # sigma_ie = 0.17
        # sigma_ii = 0.17
        sigma_ee = 0.03  # from decay=7.5
        sigma_ei = 0.04  # from decay=9.5
        sigma_ie = 0.07  # from decay=19
        sigma_ii = 0.07  # from decay=19
        # K_ee = 140
        # K_ei = 160
        # K_ie = 100
        # K_ii = 140
        # K_ee = 200
        # K_ei = 300
        # K_ie = 140
        # K_ii = 190
        K_ee = 210
        K_ei = 350
        K_ie = 140
        K_ii = 180
        nu = 10.0
        n_ext = 100
    end
end

begin
    tmax = 7u"s" # * Bump up
    tmin = 2u"s" # The transient. Simulations always begin at 0
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
                    nu,
                    n_ext)

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, tmax; populations = [:E, :I], vars = [:spike, :V],
                transient = tmin)

    ispikes = x[Population = At(:I), Var = At(:spike)]
    ispiketimes = map(eachslice(ispikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end
end

begin # * Animate
    spikes = x[Population = At(:E), Var = At(:spike)]
    rates = Dewdrop.compute_rates(spikes, 50u"ms")
    Dewdrop.animate_rates(rates, dx; filename = "critical_demo.mp4")
end

begin # * Spike raster
    spiketimes = map(eachslice(spikes, dims = Neuron)) do s
        sts = times(s)[findall(s)]
    end

    epositions = m.E.positions
    epositions = map(epositions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end
    radius = 0.2 # mm
    origin = [dx / 2, dx / 2]
    emask = map(epositions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    elocal_idxs = findall(emask)

    ipositions = m.I.positions
    ipositions = map(ipositions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end
    imask = map(ipositions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    ilocal_idxs = findall(imask)

    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "Time (ms)", ylabel
              = "Neuron index", title = "Spike raster")
    hideydecorations!(ax)

    intrvl = 5000u"ms" .. 5200u"ms"
    for (i, s) in enumerate(spiketimes[elocal_idxs])
        idxs = s .∈ [intrvl]
        scatter!(ax, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
                 markersize = 3)
    end

    ax2 = Axis(f[2, 1]; xlabel = "Time (ms)", ylabel
               = "Neuron index", title = "Spike raster (inhibitory)")
    for (i, s) in enumerate(spiketimes[ilocal_idxs])
        idxs = s .∈ [intrvl]
        scatter!(ax2, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
                 markersize = 3)
    end
    hideydecorations!(ax2)
    linkxaxes!(ax, ax2)
    display(f)
end

# begin # * Membrane potential
#     V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
#     V = set(V, 𝑡 => uconvert.(u"s", times(V)))
#     V = rectify(V, dims = 𝑡)
#     lines(V[1:2000, 970]) |> display
# end
begin # * Unit spectrum
    s = spectrum(V)
    s = mean(s, dims = Neuron)
    s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 2 .. 200]
    lines(s; axis = (; xscale = log10, yscale = log10)) |> display
end
begin # * MUA spectrum
    mdt = 3.0u"ms"
    mua = groupby(spikes, 𝑡 => Base.Fix2(Dewdrop.group_dt, mdt))
    mua = map(mua) do r
        dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", mdt)
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
begin # * Plot inter-spike interval distributions
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
