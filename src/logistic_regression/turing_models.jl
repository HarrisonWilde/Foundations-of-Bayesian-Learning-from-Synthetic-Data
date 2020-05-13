"""
    WeightedBernoulliLogit(p<:Real, w<:Real)
A univariate bernoulli logit distribution with weight w
"""
struct WeightedBernoulliLogit{T<:Real, U<:Real} <: DiscreteUnivariateDistribution
    logitp::T
	w::U
end

function Distributions.logpdf(d::WeightedBernoulliLogit{<:Real}, k::Int)
	# println("weighted: $(logpdf_bernoulli_logit(d.logitp, k))")
    return d.w * logpdf_bernoulli_logit(d.logitp, k)
end

# function Distributions.pdf(d::WeightedBernoulliLogit{<:Real}, k::Int)
#     return pdf_bernoulli_logit(d.logitp, k)
# end


"""
    BetaDBernoulliLogit(p<:Real)
A univariate bernoulli logit distribution beta-diverged
"""
struct BetaDBernoulliLogit{T<:Real, U<:Real, V<:Real} <: DiscreteUnivariateDistribution
    logitp::T
	β::U
	βw::V
end

function Distributions.logpdf(d::BetaDBernoulliLogit{<:Real}, k::Int)
    return d.βw * logpdf_betad_bernoulli_logit(d.logitp, k, d.β)
end



@model logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, w) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ WeightedBernoulliLogit.(X_real * θ, 1)
	y_synth .~ WeightedBernoulliLogit.(X_synth * θ, w)

end

@model β_logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ WeightedBernoulliLogit.(X_real * θ, 1)
	y_synth .~ BetaDBernoulliLogit.(X_synth * θ, β, βw)

end


chn = sample(β_logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw), Turing.DynamicNUTS(), 10000)
chn = sample(logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, w), Turing.NUTS(), 10000)
describe(chn, sections=:internals)

using LogDensityProblems, DynamicHMC, Turing
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

chn = sample(gdemo(1.5, 2.0), DynamicNUTS(), 200000)
