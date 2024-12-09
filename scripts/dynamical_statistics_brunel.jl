#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#

using DrWatson
DrWatson.@quickactivate
using PythonCall
using Dewdrop
using Unitful
using Statistics
using TimeseriesTools
using CairoMakie
using LinearAlgebra
using Distributed
using USydClusters
using Term
using SparseArrays
using MeanSquaredDisplacement
using Distributions

model = models.population_model.FNSPopulations
modelname = "Brunel2000"

begin # * Simulate
    N = 50000
    T = 10000.0

    m = model(N; g = 5.0, nu_hat = 2, epsilon = 0.1, D = 1.5, J = 0.1)
    X = bpsolve(m, T; populations = [:E, :I], vars = [:V])
end

begin # * Mean-squared displacement of voltage
    x = X[Population = At(:E), Var = At(:V)][:, 1]
    msd = msdist(x)[2:100]
    lines(lookup(msd, 1) .|> ustripall .|> log10, parent(msd) .|> log10;
          axis = (; aspect = DataAspect()))
end

begin
    x = X[Population = At(:E), Var = At(:V)]
    d = map(diff, eachslice(x, dims = Neuron))
    d = stack(d)
    d = d[abs.(d) .< 7]
end
begin
    f = Figure()
    ax = Axis(f[1, 1]; yscale = log10)
    hist!(d[1:100:end]; bins = 100, normalization = :pdf)
    p = fit(Normal, d[1:100:end])
    xs = -5:0.01:5
    ys = pdf.([p], xs)
    lines!(xs, ys)
    f
end
