#! /bin/bash
# -*- mode: julia -*-
#=
exec julia +1.11 -t auto --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using DrWatson
DrWatson.@quickactivate
using Dewdrop
using JLD2
Dewdrop.@preamble
set_theme!(foresight(:physics))

begin
    model = Dewdrop.models.Nonspatial
    begin
        N_e = 2000
        J_e = 0.0008 # Microsiemens
        nu = 7
        n_ext = 100
        k = 0.7
        K_ee = 72 * k
        K_ie = 28 * k
        K_ei = 88 * k
        K_ii = 39 * k
    end
end
begin
    tmax = 10u"s"
    tmin = 1u"s" # The transient. Simulations always begin at 0
    fixed_params = (; N_e, J_e, nu, n_ext, K_ee, K_ie, K_ei, K_ii)

    mua_dt = 2u"ms" # Gives mua spectrum max freq of 250 Hz

    monitors = ("E.spike",)#, "E.V", "E.input")
    stat_funcs = Dict("rate" => Dewdrop.stats.firing_rate,
                      "susceptibility" => Dewdrop.stats.susceptibility(bin = 10),
                      "spike_spectrum" => Dewdrop.stats.spike_spectrum(n_segments = 10),
                      #   "temporal_average" => Dewdrop.stats.temporal_average,
                      #   "grand_distribution" => Dewdrop.stats.grand_distribution(n_bins = 1000),
                      "mua" => Dewdrop.stats.mua(bin = ustrip(to_ms(mua_dt))))

    metadata = (; mua_dt, tmax, tmin, monitors)
end
begin# * Generate dict of parameter vectors
    sweep = (; delta = range(2.0, 7.0, length = 5))
    pnames = map(string, keys(sweep))
    pvals = stack(Iterators.product(values(sweep)...), dims = 1)
    sweep_params = Dict{String, Any}(zip(pnames, eachcol(pvals))) # Now a good shape for jax
    n_iters = length(first(values(sweep_params)))
    jax_keys = Dewdrop.jax.random.split(Dewdrop.jax.random.PRNGKey(42), n_iters)
    sweep_params["key"] = Dewdrop.numpy.array.(jax_keys) # * So that each run is independent
end
begin # * Create sweep function
    run = Dewdrop.stats.create_run(model, pydict(fixed_params), monitors,
                                   ustrip(to_ms(tmax)),
                                   ustrip(to_ms(tmin)))
    stats_run = Dewdrop.stats.create_stats_run(run, pydict(stat_funcs))
end
begin # * Run simulation
    stats, sweep_parameters = Dewdrop.stats.progress_vmap(stats_run, batch_size = 10)(pydict(sweep_params))
end
begin
    Dewdrop.stats.save("nonspatial_sweep.pickle",
                       (stats, sweep_params, fixed_params, metadata))
end
