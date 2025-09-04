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
        dx = 0.5
        rho = 20000.0
        kernel = Dewdrop.distances.ExponentialKernel
        delta = 4.0
        nu = 4.5  # External population firing rate
        n_ext = 100  # Number of external synapses per Exc. neuron

        sigma_ee = 0.075  # Width of the distance-dependent connectivity kernel (mm)
        sigma_ei = 0.095
        sigma_ie = 0.19
        sigma_ii = 0.19

        K_ee = 270
        K_ei = 350
        K_ie = 165
        K_ii = 200
    end
end

begin
    tmax = 50u"s" # * Bump up
    tmin = 10u"s" # The transient. Simulations always begin at 0
    fixed_params = (; dx, rho, kernel, n_ext,
                    sigma_ee, sigma_ei, sigma_ie, sigma_ii,
                    K_ee, K_ie, K_ei, K_ii, nu)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz
    mua_func = Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt)))

    dn = round(Int, sqrt(rho * dx^2))
    positions = Dewdrop.positions.GridPositions((dx, dx))((dn, dn))  # Maybe think about capturing this somehow
    positions = map(positions) do pos
        map(pos) do p
            p.tolist() |> convert2(Float32)
        end
    end

    # * Spatial binnin
    nbins = 20
    edges = range(0, dx, nbins + 1)
    ix = [clamp(searchsortedlast(edges, pos[1]), 1, nbins) for pos in positions]
    iy = [clamp(searchsortedlast(edges, pos[2]), 1, nbins) for pos in positions]
    bin_indices = [Int[] for _ in 1:nbins, _ in 1:nbins]
    for (neuron_idx, (bx, by)) in enumerate(zip(ix, iy))
        push!(bin_indices[bx, by], neuron_idx)
    end

    radius = 0.1 # mm
    origin = [dx / 2, dx / 2]
    mask = map(positions) do pos
        dp = abs.(pos .- origin)
        dp = min.(dp, dx .- dp)
        norm(dp) < radius
    end # scatter(positions.|> Point2f, color=mask) to check
    local_idxs = findall(mask) |> Dewdrop.numpy.asarray

    monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate
                      #   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10),
                      #   "radial_autocorrelation" => Dewdrop.stats.radial_autocorrelation(positions,
                      #                                                                    0.05))#,
                      #   "efficiency" => Dewdrop.stats.efficiency(bin_indices, 1000))
                      # "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => Dewdrop.stats.temporal_average,
                      #   "grand_distribution" => Dewdrop.stats.grand_distribution(n_bins = 1000),
                      #   "mua" => mua_func)
                      )

    metadata = (; positions, bin_indices, mua_dt, tmax, tmin, monitors)
end

begin # * Run simulation
    m = model(; fixed_params...)
    # runner = bp.DSRunner(m, monitors = monitors, numpy_mon_after_run = true)
    # runner.run(duration = duration)
    # # return {m: runner.mon[m][transient_idx:, :] for m in monitor_names}
    # m = model()
    # x = bpsolve(m, T; populations = [:E, :I], vars = [:spike])
    x = bpsolve(m, 1000; populations = [:E, :I], vars = [:spike])

    bpsolve(net::Py, time::Real; populations = [:E, :I], vars = [:V],
            transient = 500u"ms",
            inputs = nothing,
            kwargs...)
end
