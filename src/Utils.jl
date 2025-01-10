using Bootstrap
using Normalization
using StatsBase

export firingrate, cv, plotdir, connector, bootstrapaverage, bootstrapmedian,
       structurefunction, histcounts

function _preamble()
    quote
        using DrWatson
        using PythonCall
        using Unitful
        using Statistics
        using TimeseriesTools
        using CairoMakie
        using LinearAlgebra
        using Distributed
        using Term
        using SparseArrays
        using MeanSquaredDisplacement
        using Distributions
    end
end
macro preamble()
    _preamble()
end
@preamble

const connector = '&'

const UnivariateSpikeTrain = Base.typeintersect(SpikeTrain, UnivariateTimeSeries)
const MultivariateSpikeTrain = Base.typeintersect(SpikeTrain, MultivariateTimeSeries)
function firingrate(x::UnivariateSpikeTrain)
    λ = sum(x) / duration(x)
    uconvert(unit(eltype(x)) * u"Hz", λ)
end
function firingrate(X::MultivariateSpikeTrain)
    firingrate.(eachslice(X, dims = dims(X)[2:end]))
end
function cv(x::UnivariateSpikeTrain)
    ts = times(x[x])
    isis = diff(ts)
    if length(isis) < 2
        return 0.0 # No spikes, mean isi is Inf, say cv is 0
    else
        return std(isis) / mean(isis)
    end
end
function cv(X::MultivariateSpikeTrain)
    cv.(eachslice(X, dims = dims(X)[2:end]))
end

# * Bootstrap averages
_confidence(x, z = 1.96) = z * std(x) ./ sqrt(length(x)) # * 1.96 for 95% confidence interval
confidence(x, args...; dims = 1:ndims(x)) = mapslices(x -> _confidence(x, args...), x; dims)
function quartiles(X::AbstractArray; dims = 1)
    q1 = mapslices(x -> quantile(x, 0.25), X; dims)
    q2 = mapslices(x -> quantile(x, 0.5), X; dims)
    q3 = mapslices(x -> quantile(x, 0.75), X; dims)
end

function bootstrapaverage(average, x::AbstractVector{T}; confint = 0.95,
                          N = 10000)::Tuple{T, Tuple{T, T}} where {T}
    sum(!isnan, x) < 5 && return (NaN, (NaN, NaN))

    # * Estimate a sampling distribution of the average
    x = filter(!isnan, x)
    b = Bootstrap.bootstrap(nansafe(average), x, Bootstrap.BalancedSampling(N))
    μ, σ... = only(Bootstrap.confint(b, Bootstrap.BCaConfInt(confint)))
    return μ, σ
end

function bootstrapaverage(average, X::AbstractArray; dims = 1, kwargs...)
    ds = [i == dims ? 1 : Colon() for i in 1:ndims(X)]
    μ = similar(X[ds...])
    σl = similar(μ)
    σh = similar(μ)
    negdims = filter(!=(dims), 1:ndims(X)) |> Tuple
    Threads.@threads for (i, x) in collect(enumerate(eachslice(X; dims = negdims)))
        μ[i], (σl[i], σh[i]) = bootstrapaverage(average, x; kwargs...)
    end
    return μ, (σl, σh)
end
bootstrapmedian(args...; kwargs...) = bootstrapaverage(median, args...; kwargs...)
bootstrapmean(args...; kwargs...) = bootstrapaverage(mean, args...; kwargs...)

"""
Structure Function Analysis of Turbulent Flows

Generates a vector of differences at different lags, that can be used ot compute
lag-dependent statistics of the difference distribution
"""
function structurefunction(x::AbstractMatrix, τ::Int)
    Δ = @views x[(τ + 1):end, :] - x[1:(end - τ), :]
end
function structurefunction(x::AbstractVector, τ::Int)
    Δ = @views x[(τ + 1):end] - x[1:(end - τ)]
end
function structurefunction(x::RegularTimeSeries, τ::Int; kwargs...)
    structurefunction(parent(x), τ; kwargs...)
end
function structurefunction(x::RegularTimeSeries, τ::AbstractFloat)
    structurefunction(x, Int(τ ÷ step(x)); kwargs...)
end
function structurefunction(x::AbstractArray, τs)
    progressmap(τs) do τ
        structurefunction(x, τ)
    end
end
function histcounts(x, edges)
    H = fit(Histogram, x[:], edges)
    # H = normalize(H, mode = :density)
    centers = (edges[1:(end - 1)] + edges[2:end]) ./ 2
    N = sum(H.weights) .* step(edges)
    y = H.weights ./ N
    return ToolsArray(y, Dim{:bin}(centers))
end
