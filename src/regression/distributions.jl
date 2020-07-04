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



struct inputs{T, U, V}
	y::T
	X::U
	groups::V
end

struct priors{T, U}
	μₚ::T
	σₚ::T
	αₚ::T
	βₚ::T
	νₚ::T
	Σₚ::U
end


function dotmany(X, Θ, groups, nₚ)
    res = similar(groups, typeof(zero(eltype(X)) * zero(eltype(Θ))))
    @inbounds for obs in eachindex(groups)
        Θi = groups[obs]
        y = zero(eltype(res))
        for p in 1:nₚ
            y += X[obs, p] * Θ[p, Θi]
        end
        res[obs] = y
    end
    return res
end


@model β_model(
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

"""
	Hierarchical regression model with group level scales and locations
"""
@model weighted_model(
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


@model naive_model(
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

@model no_synth_model(
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
