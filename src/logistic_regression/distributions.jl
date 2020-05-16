"""
Return the logistic function computed in a numerically stable way:
``logistic(x) = 1/(1+exp(-x))``
"""
function LOGISTIC(T)
    log(one(T) / Base.eps(T) - one(T))
end
function logistic(x::T) where {T}
    LOGISTIC_T = LOGISTIC(T)
    x > LOGISTIC_T && return one(x)
    x < -LOGISTIC_T && return zero(x)
    return one(x) / (one(x) + exp(-x))
end

"""
Derivative of the logistic function
"""
function ∂logistic(z::T) where {T}
	a = exp(z) + 1
    return (1 / a) - (1 / (a ^ 2))
end


"""
	Log PDF of the Multivariate Normal distribution centred at mean = 0

A mulativariate normal distribution.
"""
function ℓpdf_MvNorm(σ::T, θ::U) where {T, U}
	return -(length(θ) * (log2π + log(abs2(σ))) + sum(@. abs2(θ / σ))) / 2
end

function ∇ℓpdf_MvNorm(σ::T, θ::U) where {T, U}
	return -θ / abs2(σ)
end


"""
    BernoulliLogit(p<:Real, w<:Real)
A bernoulli logit distribution with weight w, accepts labels 1 and -1
"""
struct BernoulliLogit{T<:Real} <: DiscreteUnivariateDistribution
    Xθ::T
end

function ℓpdf_BL(yXθ::T) where {T}
    return -log1pexp(-yXθ)
end
function Distributions.logpdf(d::BernoulliLogit{<:Real}, y::Int)
    return ℓpdf_BL(y * d.Xθ)
end

function pdf_BL(yXθ::T) where {T}
	return logistic(yXθ)
end
function Distributions.pdf(d::BernoulliLogit{<:Real}, y::Int)
    return pdf_BL(y * d.Xθ)
end

function ∇ℓpdf_BL(yX::T, θ::U) where {T, U}
	return yX' * logistic.(yX * θ)
end



"""
    wBernoulliLogit(p<:Real, w<:Real)
A univariate bernoulli logit distribution with weight w
"""
struct wBernoulliLogit{T<:Real, U<:Real} <: DiscreteUnivariateDistribution
    Xθ::T
	w::U
end

function Distributions.logpdf(d::wBernoulliLogit{<:Real}, y::Int)
    return d.w * ℓpdf_BL(y * d.Xθ)
end

function Distributions.pdf(d::wBernoulliLogit{<:Real}, y::Int)
    return d.w * pdf_BL(y * d.Xθ)
end




"""
    βBernoulliLogit(p<:Real)
A univariate bernoulli logit distribution beta-diverged
"""
struct βBernoulliLogit{T<:Real, U<:Real, V<:Real} <: DiscreteUnivariateDistribution
    Xθ::T
	β::U
	βw::V
end

function ℓpdf_βBL(Xθ::T, y::U, β::V) where {T, U, V}
    return 1 / β * pdf_BL(y * Xθ) ^ β - 1 / (β + 1) * (logistic(Xθ) ^ (β + 1) + logistic(-Xθ) ^ (β + 1))
end
function Distributions.logpdf(d::βBernoulliLogit{<:Real}, y::Int)
    return d.βw * ℓpdf_βBL(d.Xθ, y, d.β)
end

function yXβ(yX::T, β::U, θ::V) where {T, U, V}
    eyXθ = exp.(-yX * θ)
    return yX' * (eyXθ ./ (1 .+ eyXθ) .^ β)
end
function ∇ℓpdf_βBL(yX::T, β::U, θ::V) where {T, U, V}
    return yXβ(yX, β + 1, θ) - yXβ(yX, β + 2, θ) - yXβ(-yX, β + 2, θ)
end


@model β_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ βBernoulliLogit.(X_synth * θ, β, βw)

end

@model weighted_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, w) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ wBernoulliLogit.(X_synth * θ, w)

end

@model naive_model(X_real, X_synth, y_real, y_synth, θ_dim, σ) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ Main.BernoulliLogit.(X_synth * θ)

end

@model no_synth_model(X_real, y_real, θ_dim, σ) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)

end
