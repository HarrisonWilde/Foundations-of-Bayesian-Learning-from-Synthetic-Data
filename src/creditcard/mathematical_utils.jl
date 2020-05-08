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
function ∂logistic(z::Float64)
	a = exp(z) + 1
    return (1 / a) - (1 / (a ^ 2))
end


"""
	Log PDF of the Multivariate Normal distribution centred at mean = 0

A mulativariate normal distribution.
"""
function logpdf_centred_mvnormal(σ::T, θ::U) where {T, U}
	return -(length(θ) * (log2π + log(abs2(σ))) + sum(@. abs2(θ / σ))) / 2
end


"""
    Log PDF of the logit-parameterised bernoulli distribution

A univariate bernoulli logit distribution.
"""
function logpdf_bernoulli_logit(z::T, y::U) where {T, U}
    return y * z - log1pexp(z)
end


"""
    PDF of the logit-parameterised bernoulli distribution; z = x * θ

A univariate bernoulli logit distribution.
"""
function pdf_bernoulli_logit(z::T, y::U) where {T, U}
	y * logistic(z) + (1 - y) * (1 - logistic(z))
end


"""
	Beta-Divergence applied to the Log PDF of the logit-parameterised bernoulli distribution

A univariate bernoulli logit distribution with the beta divergence applied.
"""
function logpdf_betad_bernoulli_logit(z::T, y::U, β::V) where {T, U, V}
	@. (1.0 / β) * (
		pdf_bernoulli_logit(z, y)
	) ^ β - (1.0 / (β + 1.0)) * (
		logistic(z) ^ (β + 1.0)
		+ (1.0 - logistic(z)) ^ (β + 1.0)
	)
end
