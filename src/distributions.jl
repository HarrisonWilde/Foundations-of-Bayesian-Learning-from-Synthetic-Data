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
    Normal distribution
A univariate Normal distribution.
"""
struct MyNormal{T<:Real, U<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
end


function ℓpdf_N(μ, σ, y::Array)
	return -length(y) * log(σ) - (length(y) / 2) * log2π - sum(abs2.(y .- μ)) / (2 * abs2(σ))
end
function ℓpdf_N(μ, σ, y::Real)
	return -log(σ) - 0.5 * (log2π + abs2((y - μ) / σ))
end
function Distributions.logpdf(d::MyNormal{<:Real}, y::Real)
    return ℓpdf_N(d.μ, d.σ, y)
end

function pdf_N(μ, σ, y)
	return exp(-abs2((y - μ) / σ) / 2) * invsqrt2π / σ
end
function Distributions.pdf(d::MyNormal{<:Real}, y::Real)
    return pdf_N(d.μ, d.σ, y)
end

function cdf_N(μ, σ, y)
	return erfc(-(y - μ) / (σ * sqrt2)) / 2
end
# function Distributions.cdf(d::MyNormal{<:Real}, y::real)
# 	return cdf_N(d.μ, d.σ, y)
# end


"""
    w-Normal distribution
A univariate weighted Normal distribution.
"""
struct wNormal{T<:Real, U<:Real, V<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
    w::V
end

function Distributions.logpdf(d::wNormal{<:Real}, y::Real)
    return d.w * ℓpdf_N(d.μ, d.σ, y)
end

function Distributions.pdf(d::wNormal{<:Real}, y::Real)
    return d.w * pdf_N(d.μ, d.σ, y)
end



"""
	β-Normal distribution
A univariate Normal distribution using β divergence for updating
"""
struct βNormal{T<:Real, U<:Real, V<:Real, W<:Real, X<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
	βw::V
	β::W
	∫term::X
end

function int_term(σ², β)
	1 / (((2π) ^ (β / 2)) * ((1 + β) ^ 1.5) * (σ² ^ (β / 2)))
end
function ℓpdf_βN(μ, σ, β, y)
	return @. (1 / β) * pdf_N(μ, σ, y) ^ β - int_term(abs2(σ), β)
end
function ℓpdf_βN(μ, σ, β, ∫term, y)
	return @. (1 / β) * pdf_N(μ, σ, y) ^ β - ∫term
end
function Distributions.logpdf(d::βNormal{<:Real}, y::Real)
    return d.βw * ℓpdf_βN(d.μ, d.σ, d.β, d.∫term, y)
end



"""
    Normal-Laplace distribution
A univariate Normal (symmetric-)Laplace convolved distribution using β divergence for updating
"""
struct NormalLaplace{T<:Real, U<:Real, V<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
    λ::V
end


function ℓpdf_NL(μ, σ, λ, y)
    return log(pdf_NL(μ, σ, λ, y))
end
function Distributions.logpdf(d::NormalLaplace{<:Real}, y::Real)
    return ℓpdf_NL(d.μ, d.σ, d.λ, y)
end

function pdf_NL(μ, σ, λ, y)
    return (1 / 4λ) * (
        exp((μ - y) / λ + abs2(σ) / (2 * abs2(λ))) *
        (1 + erf((y - μ) / (√2 * σ) - σ / (√2 * λ))) +
        exp((y - μ) / λ + abs2(σ) / (2 * abs2(λ))) *
        (1 - erf((y - μ) / (√2 * σ) + σ / (√2 * λ)))
    )
end

function Distributions.pdf(d::NormalLaplace{<:Real}, y::Real)
    return pdf_NL(d.μ, d.σ, d.λ, y)
end



@model β_gaussian_model(y_real, y_synth, βw, β, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

	∫term = int_term(σ², β)
	if length(y_real) > 0
		y_real .~ Distributions.Normal(μ, √σ²)
	end
	if length(y_synth) > 0
		y_synth .~ βNormal(μ, √σ², βw, β, ∫term)
	end

end

@model weighted_gaussian_model(y_real, y_synth, w, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

	if length(y_real) > 0
		y_real .~ Distributions.Normal(μ, √σ²)
	end
	if length(y_synth) > 0
		y_synth .~ wNormal(μ, √σ², w)
	end

end

@model naive_gaussian_model(y_real, y_synth, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

    y_real .~ Distributions.Normal(μ, √σ²)
    y_synth .~ Distributions.Normal(μ, √σ²)

end

@model no_synth_gaussian_model(y_real, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

    y_real .~ Distributions.Normal(μ, √σ²)

end

@model β_all_gaussian_model(y_real, y_synth, βw, β, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

    ∫term = int_term(σ², β)
    y_real .~ βNormal(μ, √σ², βw, β, ∫term)
    y_synth .~ βNormal(μ, √σ², βw, β, ∫term)

end

@model noise_aware_gaussian_model(y_real, y_synth, λ, αₚ, βₚ, μₚ, σₚ) = begin

    σ² ~ InverseGamma(αₚ, βₚ)
    μ ~ Distributions.Normal(μₚ, σₚ * √σ²)

    y_real .~ Distributions.Normal(μ, √σ²)
    y_synth .~ NormalLaplace(μ, √σ², λ)

end

@model β_regression_model(
	y_real, X_real, groups_real,
	y_synth, X_synth, groups_synth,
	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
	nₚ, nₛ, β, βw) = begin

	ασ² ~ InverseGamma(αₚ, βₚ)
	βσ² ~ InverseGamma(αₚ, βₚ)
	μθ ~ Normal(μₚ, σₚ)
	Σ ~ InverseWishart(νₚ, Σₚ)

	Θ ~ filldist(MvNormal([μθ for _ in 1:nₚ], Σ), nₛ)
	σ² ~ filldist(InverseGamma(ασ², βσ²), nₛ)

	y_real .~ Normal.(
		dotmany(X_real, Θ, groups_real, nₚ),
		# dot.(eachrow(X_real), eachcol(Θ[:, groups_real])),
		sqrt.(σ²[groups_real])
	)
	∫terms = int_term.(σ², β)
	y_synth .~ βNormal.(
		dotmany(X_synth, Θ, groups_synth, nₚ),
		# dot.(eachrow(X_synth), eachcol(Θ[:, groups_synth])),
		sqrt.(σ²[groups_synth]),
		βw,
		β,
		∫terms[groups_synth]
	)

end

@model weighted_regression_model(
	y_real, X_real, groups_real,
	y_synth, X_synth, groups_synth,
	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
	nₚ, nₛ, w) = begin

	ασ² ~ InverseGamma(αₚ, βₚ)
	βσ² ~ InverseGamma(αₚ, βₚ)
	μθ ~ Normal(μₚ, σₚ)
	Σ ~ InverseWishart(νₚ, Σₚ)

	Θ ~ filldist(MvNormal([μθ for _ in 1:nₚ], Σ), nₛ)
	σ² ~ filldist(InverseGamma(ασ², βσ²), nₛ)

	y_real .~ Normal.(
		dot.(eachrow(X_real), eachcol(Θ[:, groups_real])),
		# dotmany(X_real, Θ, groups_real, nₚ),
		sqrt.(σ²[groups_real])
	)
	y_synth .~ wNormal.(
		dot.(eachrow(X_synth), eachcol(Θ[:, groups_synth])),
		# dotmany(X_synth, Θ, groups_synth, nₚ),
		sqrt.(σ²[groups_synth]),
		w
	)

end

@model naive_regression_model(
	y_real, X_real, groups_real,
	y_synth, X_synth, groups_synth,
	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
	nₚ, nₛ) = begin

	ασ² ~ InverseGamma(αₚ, βₚ)
	βσ² ~ InverseGamma(αₚ, βₚ)
	μθ ~ Normal(μₚ, σₚ)
	Σ ~ InverseWishart(νₚ, Σₚ)

	Θ ~ filldist(MvNormal([μθ for _ in 1:nₚ], Σ), nₛ)
	σ² ~ filldist(InverseGamma(ασ², βσ²), nₛ)

	y_real .~ Normal.(
		dot.(eachrow(X_real), eachcol(Θ[:, groups_real])),
		# dotmany(X_real, Θ, groups_real, nₚ),
		sqrt.(σ²[groups_real])
	)
	y_synth .~ Normal.(
		dot.(eachrow(X_synth), eachcol(Θ[:, groups_synth])),
		# dotmany(X_synth, Θ, groups_synth, nₚ),
		sqrt.(σ²[groups_synth])
	)

end

@model no_synth_regression_model(
	y_real, X_real, groups_real,
	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
	nₚ, nₛ) = begin

	ασ² ~ InverseGamma(αₚ, βₚ)
	βσ² ~ InverseGamma(αₚ, βₚ)
	μθ ~ Normal(μₚ, σₚ)
	Σ ~ InverseWishart(νₚ, Σₚ)

	Θ ~ filldist(MvNormal([μθ for _ in 1:nₚ], Σ), nₛ)
	σ² ~ filldist(InverseGamma(ασ², βσ²), nₛ)

	y_real .~ Normal.(
		dot.(eachrow(X_real), eachcol(Θ[:, groups_real])),
		# dotmany(X_real, Θ, groups_real, nₚ),
		sqrt.(σ²[groups_real])
	)

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


@model β_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ βBernoulliLogit.(X_synth * θ, β, βw)

end

@model weighted_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, w) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ wBernoulliLogit.(X_synth * θ, w)

end

@model naive_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)
	y_synth .~ Main.BernoulliLogit.(X_synth * θ)

end

@model no_synth_logistic_model(X_real, y_real, θ_dim, σ) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ Main.BernoulliLogit.(X_real * θ)

end
