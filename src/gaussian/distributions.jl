"""
	Normal distribution
A univariate Normal distribution.
"""
struct Normal{T<:Real, U<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
end


function ℓpdf_N(μ, σ, y::Array)
	return -length(y) * log(σ) - (length(y) / 2) * log2π - sum(abs2.(y .- μ)) / (2 * abs2(σ))
end
function ℓpdf_N(μ, σ, y::Real)
	return -log(σ) - 0.5 * (log2π + abs2((y - μ) / σ))
end
function Distributions.logpdf(d::Main.Normal{<:Real}, y::Real)
    return ℓpdf_N(d.μ, d.σ, y)
end

function pdf_N(μ, σ, y)
	return exp(-abs2((y - μ) / σ) / 2) * invsqrt2π / σ
end
function Distributions.pdf(d::Main.Normal{<:Real}, y::Real)
    return pdf_N(d.μ, d.σ, y)
end


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
function ℓpdf_βN(μ, σ, β, ∫term, y)
	return @. (1 / β) * pdf_N(μ, σ, y) ^ β - ∫term
end
function Distributions.logpdf(d::βNormal{<:Real}, y::Real)
    return d.βw * ℓpdf_βN(d.μ, d.σ, d.β, d.∫term, y)
end


"""
	Normal-Laplace distribution
A univariate Normal Laplace convolved distribution using β divergence for updating
"""
struct NormalLaplace{T<:Real, U<:Real, V<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::U
	λ::V
end


function ℓpdf_NL(μ, σ, α, λ, y)
	return log(pdf_NL(μ, σ, α, λ, y))
end
function Distributions.logpdf(d::NormalLaplace{<:Real}, y::Real)
    return ℓpdf_NL(d.μ, d.σ, d.λ, y)
end

function pdf_NL(μ, σ, λ, y)
	return (λ / 4) * erfc((λ * σ) / √2 - (y - μ) / (√2 * σ)) * (
		exp(0.5λ * (-2y + 2μ + λ * abs2(σ))) +
		exp(0.5λ * (2y - 2μ + λ * abs2(σ)))
	)
end
function Distributions.pdf(d::NormalLaplace{<:Real}, y::Real)
    return pdf_NL(d.μ, d.σ, d.λ, y)
end





@model β_model(y_real, y_synth, βw, β, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

	∫term = int_term(σ², β)
    y_real .~ Main.Normal(μ, √σ²)
	y_synth .~ βNormal(μ, √σ², βw, β)

end

@model weighted_model(y_real, y_synth, w, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

    y_real .~ Main.Normal(μ, √σ²)
	y_synth .~ wNormal(μ, √σ², w)

end

@model naive_model(y_real, y_synth, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

    y_real .~ Main.Normal(μ, √σ²)
	y_synth .~ Main.Normal(μ, √σ²)

end

@model no_synth_model(y_real, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

    y_real .~ Main.Normal(μ, √σ²)

end

@model β_all_model(y_real, y_synth, βw, β, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

	∫term = int_term(σ², β)
    y_real .~ βNormal(μ, √σ², βw, β, ∫term)
	y_synth .~ βNormal(μ, √σ², βw, β, ∫term)

end

@model noise_aware_model(y_real, y_synth, λ, αₚ, βₚ, μₚ, σₚ) = begin

	σ² ~ InverseGamma(αₚ, βₚ)
	μ ~ Main.Normal(μₚ, σₚ * √σ²)

    y_real .~ Main.Normal(μ, √σ²)
	y_synth .~ NormalLaplace(μ, √σ², λ)

end
