export firingrate, cv

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
