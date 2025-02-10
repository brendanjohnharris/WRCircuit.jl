#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using Dewdrop
Dewdrop.@preamble
set_theme!(foresight(:physics))

model = models.FNScircuit
modelname = "FNScircuit"

begin # * Simulate
    m = model(; rho = 30000, nu = 100.0, n_ext = 10, J_e = 0.0005, zeta = 2)
    m.to_dict()
end
begin
    T = 5500.0
    cue = Dewdrop.numpy.array(m.E.V) |> convert2(Vector)
    Cue = reshape(cue, convert2(Vector)(m.E.size)...)
    Cue[:] .= 0.0
    Cue[1:4, 1:4] .= 0.5 # Cue the bottom corner
    # cue = cue)
    # cue = repeat(cue, 1, Int(T÷convert2(Float32)(Dewdrop.brainpy.share["dt"])))
    X = bpsolve(m, T; populations = [:E, :I], vars = [:spike, :V, :input, :g_K])#, inputs = [("Ein.input", cue)])

    V = X[Var = At(:V)]
    g_K = X[Var = At(:g_K)]
    input = X[Var = At(:input)]
    spikes = X[Var = At(:spike)]
    pop = :E # Excitatory
    lines(input[1][1:1000, 1])
end
begin # * Animate spikes
    T = 0u"ms" .. 1000u"ms"
    dt = 1u"ms"
    N = m.E.size |> convert2(Vector)
    domain = m.E.embedding.domain |> convert2(Vector)
    dx = domain ./ N
    xs = range.(0 .+ dx / 2, domain .- dx / 2, N)

    f = Figure()
    ax = Axis(f[1, 1])
    x = spikes[1][𝑡 = T] # Excitatory spikes
    x = sum.(coarsegrain(x, dt)) # Bin over time
    x = set(x, 𝑡 => mean.(times(x))) |> ustripall # To unitless seconds
    h = Observable(parent(x[1, :]))
    H = lift(h) do h
        reshape(h, N...)
    end
    _t = Observable(first(times(x)))
    colorrange = extrema(x)
    heatmap!(ax, xs..., H; colorrange, colormap = :binary)
    rm("network.mp4"; force = true)
    record(f, "network.mp4", eachindex(times(x))) do i
        h[] = parent(x[i, :])
        _t[] = times(x)[i]
    end
    @info "Animation saved to `./network.mp4`"
    f
end
begin # Spike raster
    f = Figure(size = (1920, 480))
    gs = subdivide(f, 1, 4)

    T = 500u"ms" .. 1000u"ms"

    # First do spike raster with firing rate
    ax = Axis(gs[1][1, 1])
    x = spiketimes(spikes[Population = At(pop)][𝑡 = T][:, 1:100])
    spikeraster!(ax, 1:length(x), x; markersize = 5) # Switch to clustering
    f
end
begin # * Voltage traces
    ax = Axis(gs[1][2, 1])
    x = V[Population = At(pop)][𝑡 = T][:, [1]]
    map(eachslice(x; dims = 2)) do _x
        lines!(ax, ustripall(_x))
    end
    f
end
begin # * Synaptic input distribution
    τ = 1 # ms
    syn = coarsegrain(input[Population = At(pop)], τ) .|> sum
    syn = Iterators.flatten(syn) |> collect
    ax = Axis(gs[2]; yscale = log10, limits = (nothing, (1e-8, 1)))
    ziggurat!(syn; bins = 50, normalization = :pdf, color = Cycled(1))

    D = fit(Normal, syn)
    xs = range(extrema(syn)...; length = 500)
    lines!(ax, xs, pdf.(D, xs); color = Cycled(2), linestyle = :dash)

    D = fit(Stable, syn)
    D = Stable(D.α, 0, D.σ, D.μ)
    lines!(ax, xs, pdf.(D, xs); color = Cycled(4), linestyle = :dash)
    f
end
begin # Some dynamical statistics (firing rate, fano factor, etc...)
    stats = [firingrate, cv] #, fanofactor]
    for (i, s) in enumerate(stats)
        y = s(spikes[Population = At(pop)]) |> ustripall |> collect
        nzero = sum(y .<= 0) / length(y)
        nzero = round(Int, nzero * 100)
        ax = Axis(gs[3][i, 1]; title = "$s: $nzero% zeros")
        y = y[y .> 0]
        ziggurat!(ax, y, label = string(s); color = Cycled(i))
    end
    f
end
begin # * Power spectrum of voltage fluctuations
    x = V[Population = At(pop)][:, 1:100:end]
    x = set(x, 𝑡 => ustripall(times(x)) ./ 1000) # To unitless seconds
    x = set(x, Neuron => 1:size(x, 2))
    x = x .- mean(x, dims = 1)
    y = progressmap(x -> spectrum(x, 1), eachslice(x; dims = 2)) |> stack
    # y = mean(y, dims = 2)
    # y = dropdims(y, dims = 2)
    ax = Axis(gs[4])
    spectrumplot!(ax, y[2:end, :])
    f |> display
end
