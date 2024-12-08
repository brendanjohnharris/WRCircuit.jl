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

model = models.population_model.FNSPopulations
modelname = "Brunel2000"

begin # * Simulate
    N = 50000
    T = 1000.0

    m = model(N; g = 5.0, nu_hat = 2, epsilon=0.1, D=1.5, J=0.1)
    x = bpsolve(m, T; populations = [:E, :I], vars = [:V])
end

begin # * Mean-squared displacement of voltage


end
