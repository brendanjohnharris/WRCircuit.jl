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

begin
    model = Dewdrop.models.Spatial
    begin # FNS parameters
        rho = 20000
        dx = 0.5
        # sigma_ee = 0.07
        # sigma_ei = 0.095
        # sigma_ie = 0.17
        # sigma_ii = 0.17
        sigma_ee = 0.06  # from decay=7.5
        sigma_ei = 0.07  # from decay=9.5
        sigma_ie = 0.14  # from decay=19
        sigma_ii = 0.14  # from decay=19
        # K_ee = 140
        # K_ei = 160
        # K_ie = 100
        # K_ii = 140
        # K_ee = 200
        # K_ei = 300
        # K_ie = 140
        # K_ii = 190
        K_ee = 260
        K_ei = 340
        K_ie = 225
        K_ii = 290
        nu = 10.0
        n_ext = 100
        Delta_g_K = 0.002
    end
end

begin
    tmax = 10u"s" # * Bump up
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
                    n_ext,
                    Delta_g_K)

    # monitors = ["E.spike", ("E.input", local_idxs)] |> pytuple
    # stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
    #                   "susceptibility" => Dewdrop.stats.susceptibility(bin = 10))
end

begin # * Run simulation
    m = model(; fixed_params...)
    x = bpsolve(m, tmax; populations = [:E], vars = [:spike, :V, :input],
                transient = tmin)
end

begin # * Animate
    spikes = x[Population = At(:E), Var = At(:spike)]
    rates = Dewdrop.compute_rates(spikes, 50u"ms")
    Dewdrop.animate_rates(rates, dx; filename = "critical_demo.mp4")
end

# begin # * Spike raster
#     ispikes = x[Population = At(:I), Var = At(:spike)]
#     ispiketimes = map(eachslice(ispikes, dims = Neuron)) do s
#         sts = times(s)[findall(s)]
#     end

#     spiketimes = map(eachslice(spikes, dims = Neuron)) do s
#         sts = times(s)[findall(s)]
#     end

#     epositions = m.E.positions
#     epositions = map(epositions) do pos
#         map(pos) do p
#             p.tolist() |> convert2(Float32)
#         end
#     end
#     radius = 0.2 # mm
#     origin = [dx / 2, dx / 2]
#     emask = map(epositions) do pos
#         dp = abs.(pos .- origin)
#         dp = min.(dp, dx .- dp)
#         norm(dp) < radius
#     end # scatter(positions.|> Point2f, color=mask) to check
#     elocal_idxs = findall(emask)

#     ipositions = m.I.positions
#     ipositions = map(ipositions) do pos
#         map(pos) do p
#             p.tolist() |> convert2(Float32)
#         end
#     end
#     imask = map(ipositions) do pos
#         dp = abs.(pos .- origin)
#         dp = min.(dp, dx .- dp)
#         norm(dp) < radius
#     end # scatter(positions.|> Point2f, color=mask) to check
#     ilocal_idxs = findall(imask)

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Time (ms)", ylabel
#               = "Neuron index", title = "Spike raster")
#     hideydecorations!(ax)

#     intrvl = 5000u"ms" .. 5200u"ms"
#     for (i, s) in enumerate(spiketimes[elocal_idxs])
#         idxs = s .∈ [intrvl]
#         scatter!(ax, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
#                  markersize = 3)
#     end

#     ax2 = Axis(f[2, 1]; xlabel = "Time (ms)", ylabel
#                = "Neuron index", title = "Spike raster (inhibitory)")
#     for (i, s) in enumerate(spiketimes[ilocal_idxs])
#         idxs = s .∈ [intrvl]
#         scatter!(ax2, ustripall(s[idxs]), i * ones(sum(idxs)), color = :black,
#                  markersize = 3)
#     end
#     hideydecorations!(ax2)
#     linkxaxes!(ax, ax2)
#     display(f)
# end

# begin # * Membrane potential
#     V = x[Population = At(:E), Var = At(:V)][1:10:end, :]
#     V = set(V, 𝑡 => uconvert.(u"s", times(V)))
#     V = rectify(V, dims = 𝑡)
#     lines(V[1:2000, 970]) |> display
# end
# begin # * Mean each V, zero out the 10ms around each spike
#     V = x[Population = At(:E), Var = At(:V)]
#     V = deepcopy(V[1:10:end, :])
#     V = set(V, 𝑡 => uconvert.(u"s", times(V)))
#     V = rectify(V, dims = 𝑡)
#     V̂ = V .- mean(V, dims = 𝑡)
#     map(eachslice(V̂, dims = Neuron), eachslice(spikes, dims = Neuron)) do v, s
#         sidxs = times(s[s])
#         ints = map(sidxs) do t
#             (t - 10u"ms") .. (t + 10u"ms")
#         end
#         for int in ints
#             v[𝑡 = int] .= 0.0
#         end
#     end
# end
# begin # * Average unit spectrum
#     _s = spectrum(V̂)
#     s = median(_s, dims = Neuron)
#     s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 2 .. 100]
#     # s = _s[2:end, 2000] |> ustripall

#     ls = Dewdrop.log10spectrum(s)
#     params = fit_oneoneff(ls; n_peaks = 2, w = 10)
#     params = fit_oneoneff(ls, params)

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Log frequency", ylabel
#               = "Log power (a.u.)", title = "MUA spectrum with fit")
#     lines!(ax, ls; color = :blue)
#     lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params); color = :red)
#     display(f)
# end
# begin # * Mean membrane potential in a local patch
#     idxs = [1:10, 1:10]
#     N = lookup(V, Neuron) |> length |> sqrt |> Int
#     localV = reshape(V, (size(V, 1), N, N))
#     localV = localV[:, idxs...]
#     localV = mean(localV, dims = (2, 3))
#     localV = ToolsArray(vec(localV), dims(V, 𝑡))
#     lines(localV[1:5000]) |> display
#     hist(localV, bins = 200) |> display
# end

# begin # * Spectrum of local membrane potential mean
#     V = x[Population = At(:E), Var = At(:V)]
#     V = set(V, 𝑡 => convert2(u"s", times(V)))
#     N = lookup(V, Neuron) |> length |> sqrt |> Int
#     _V = reshape(V, (size(V, 1), N, N))

#     block_size = 10
#     num_row_blocks = div(size(_V, 2), block_size)
#     num_col_blocks = div(size(_V, 3), block_size)

#     idxs = [((i * block_size + 1):((i + 1) * block_size),
#              (j * block_size + 1):((j + 1) * block_size))
#             for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)]

#     LFP = map(idxs) do (i, j)
#         m = _V[:, i, j] # * Get local patch
#         m = mean(m, dims = (2, 3))
#         m = ToolsArray(vec(m), dims(V, 𝑡))
#     end
#     LFP = ToolsArray(LFP[:], Obs(1:length(LFP))) |> stack
#     sLFP = spectrum(LFP .- mean(LFP, dims = 𝑡), 0.5u"s")

#     s = mean(sLFP, dims = Obs) |> ustripall
#     s = dropdims(s, dims = Obs)[𝑓 = 0.5 .. 10000]
#     lines(s; axis = (; xscale = log10, yscale = log10)) |>
#     display
# end

# begin # * Mean spectrum fit
#     ls = logsample(s[3:end])
#     params = fit_oneoneff(ls; n_peaks = 2, w = 2)
#     params = fit_oneoneff(ls, params)

#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Frequency (Hz)", ylabel
#               = "Power (a.u.)", title = "MUA spectrum with fit")
#     lines!(ax, ls; color = :blue)
#     lines!(ax, lookup(ls, 1), oneoneff(lookup(ls, 1), params); color = :red),
#     display(f)
# end

# begin # * MUA spectrum
#     mdt = 2.0u"ms"
#     mua = groupby(spikes, 𝑡 => Base.Fix2(Dewdrop.group_dt, mdt))
#     mua = map(mua) do r
#         dropdims(sum(r, dims = 𝑡), dims = 𝑡) ./ uconvert(u"s", mdt)
#     end |> stack
#     mua = permutedims(mua, (𝑡, Neuron))
#     mua = rectify(mua, dims = 𝑡)
#     N = lookup(mua, Neuron) |> length |> sqrt |> Int
#     ts = dims(mua, 𝑡)
#     mua = reshape(mua, (size(mua, 1), N, N))

#     # Calculate number of complete blocks that fit in each dimension
#     block_size = 10
#     num_row_blocks = div(size(mua, 2), block_size)
#     num_col_blocks = div(size(mua, 3), block_size)

#     idxs = [((i * block_size + 1):((i + 1) * block_size),
#              (j * block_size + 1):((j + 1) * block_size))
#             for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)]

#     muas = map(idxs) do (i, j)
#         m = mua[:, i, j] # * Get local patch
#         m = mean(m, dims = (2, 3))
#         m = ToolsArray(vec(m), ts)
#         m = set(m, 𝑡 => uconvert.(u"s", times(m)))
#         m = rectify(m, dims = 𝑡)
#         return spectrum(m)
#     end

#     muas = mean(muas)
#     lines(ustripall(muas)[𝑓 = 1 .. 200]; axis = (; xscale = log10, yscale = log10)) |>
#     display
# end
# begin # * Plot inter-spike interval distributions
#     isis = map(spiketimes) do sts
#         diff(sts)
#     end
#     f = Figure()
#     ax = Axis(f[1, 1]; xlabel = "Inter-spike interval (ms)",
#               ylabel = "Density", limits = ((0, 100), nothing))
#     hist!(ax, Iterators.flatten(isis) |> collect |> ustrip, normalization = :pdf,
#           bins = range(0.0, 1000, step = 5))
#     display(f)
# end

begin # * Input trace
    input = x[Population = At(:E), Var = At(:input)][:, 900]
    input = set(input, 𝑡 => uconvert.(u"s", times(input)))
    input = rectify(input, dims = 𝑡)
    lines(input[1:9000]) |> display
end
begin # * Input distribution
    input = x[Population = At(:E), Var = At(:input)]
    input = log10.(input[input .> 0.1])
    hist(input[:], bins = 100, axis = (; yscale = log10))
end
# begin # * Input spectrum
#     input = x[Population = At(:E), Var = At(:input)]
#     input = set(input, 𝑡 => uconvert.(u"s", times(input)))
#     input = rectify(input, dims = 𝑡)
#     _s = spectrum(input .- mean(input, dims = 𝑡), 0.5)
#     s = mean(_s, dims = Neuron)
#     s = ustripall(dropdims(s, dims = Neuron))[𝑓 = 0.1 .. 200]
#     lines(s; axis = (; xscale = log10, yscale = log10)) |> display
# end
# begin # * MSD of inputs
#     input = x[Population = At(:E), Var = At(:input)]
#     msd = msdist(input, 1:100)
#     msd = mean(msd, dims = Neuron) |> ustripall
#     msd = dropdims(msd, dims = Neuron)
#     lines(msd, axis = (; xscale = log10, yscale = log10)) |> display
# end

# begin # * Increments of voltage
#     dV = diff(V, dims = 𝑡)
#     dV[dV .< 10] .= NaN
#     lines(dV[1:1000, 1])
# end

begin # * Plot the MSD and power spectrum, with fits, of the LFP, membrane potential, and input traces
    begin # * Membrane potential
        V = x[Population = At(:E), Var = At(:V)]
        V = set(V, 𝑡 => convert2(u"s", times(V)))
    end
    begin # * LFP
        N = lookup(V, Neuron) |> length |> sqrt |> Int
        _V = reshape(V, (size(V, 1), N, N))

        block_size = 10
        num_row_blocks = div(size(_V, 2), block_size)
        num_col_blocks = div(size(_V, 3), block_size)

        idxs = [((i * block_size + 1):((i + 1) * block_size),
                 (j * block_size + 1):((j + 1) * block_size))
                for i in 0:(num_row_blocks - 1), j in 0:(num_col_blocks - 1)]

        LFP = map(idxs) do (i, j)
            m = _V[:, i, j] # * Get local patch
            m = mean(m, dims = (2, 3))
            m = ToolsArray(vec(m), dims(V, 𝑡))
        end
        LFP = ToolsArray(LFP[:], Obs(1:length(LFP))) |> stack
    end
    begin # * Inputs
        input = x[Population = At(:E), Var = At(:input)]
        input = set(input, 𝑡 => convert2(u"s", times(V)))
    end
end

function fit_spectrum(s; components, peaks, f_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑓)] |> Tuple
    s = s[𝑓 = f_range] |> ustripall
    s = mean(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    _s = logsample(s)
    m = fit(MAPPLE, _s; components, peaks)
    fit!(m, _s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s, _s)
end
function fit_mad(s; components, peaks, tau_range)
    negdims = [i for i in 1:ndims(s) if i != dimnum(s, 𝑡)] |> Tuple
    s = s[𝑡 = tau_range] |> ustripall
    s = mean(s, dims = negdims)
    s = dropdims(s, dims = negdims)
    # _s = logsample(s)
    m = fit(MAPPLE, s; components, peaks)
    fit!(m, s)
    fitted_s = predict(m, s)
    return (; m, s, fitted_s)
end
begin # * Calculate spectra and MAD
    vars = (; V, LFP, input)
    spectra = map(Chart(ProgressLogger()), vars) do v
        spectrum(v .- mean(v, dims = 𝑡), 0.5u"Hz")
    end
    mads = map(vars) do v
        madev(v, round.(Int, logrange(10, 10000, length = 100) |> unique))
    end
end
begin # * Fits
    f_range = 6u"Hz" .. 1000u"Hz"
    tau_range = 0u"s" .. 1u"s"
    fit_spectra = map(spectra) do s
        fit_spectrum(s; components = 1, peaks = 1, f_range)
    end
    fit_mads = map(mads) do m
        fit_mad(m; components = 2, peaks = 0, tau_range)
    end
end
begin
    f = SixPanel()
    gs = permutedims(subdivide(f, 3, 2), (2, 1))

    axs = map(enumerate(keys(vars))) do (i, v)
        s = fit_spectra[v].s
        _s = fit_spectra[v]._s
        fitted_s = fit_spectra[v].fitted_s
        m = fit_spectra[v].m

        ax = Axis(gs[i, 1]; xscale = log10, yscale = log10, title = string(v),
                  xlabel = "Frequency (Hz)", ylabel = "PSD (a.u.)")
        lines!(ax, s; color = :black, alpha = 0.4)
        scatter!(ax, _s; color = :black, markersize = 10)
        lines!(ax, fitted_s; color = :red)
        text = m.params.components.β |> last
        text = "β = $(round(text, digits = 2))"
        text!(ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
              align = (:left, :bottom))
        return ax
    end
    # linkaxes!(axs...)

    axs = map(enumerate(keys(vars))) do (i, v)
        s = fit_mads[v].s
        _s = s#fit_mads[v]._s
        fitted_s = fit_mads[v].fitted_s
        m = fit_mads[v].m

        ax = Axis(gs[i, 2]; xscale = log10, yscale = log10, title = string(v),
                  xlabel = "Frequency (Hz)", ylabel = "MSD (a.u.)")
        lines!(ax, s; color = :black, alpha = 0.4)
        scatter!(ax, _s; color = :black, markersize = 10)
        lines!(ax, fitted_s; color = :red)
        text = m.params.components.β |> first
        text = "β = $(round(text, digits = 2))"
        text!(ax, 0.1, 0.1; text, fontsize = 16, space = :relative,
              align = (:left, :bottom))
        return ax
    end
    # linkaxes!(axs...)
    display(f)
end
