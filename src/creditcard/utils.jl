using SpecialFunctions
using StatsFuns: log1pexp, log2π

"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
end



const LOGISTIC_64 = log(Float64(1)/eps(Float64) - Float64(1))
const LOGISTIC_32 = log(Float32(1)/eps(Float32) - Float32(1))

"""
Return the logistic function computed in a numerically stable way:
``logistic(x) = 1/(1+exp(-x))``
"""
function logistic(x::Float64)
	x > LOGISTIC_64  && return one(x)
	x < -LOGISTIC_64 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
function logistic(x::Float32)
	x > LOGISTIC_32  && return one(x)
	x < -LOGISTIC_32 && return zero(x)
	return one(x) / (one(x) + exp(-x))
end
logistic(x) = logistic(float(x))
ℓog = logistic

"""
Derivative of the logistic function
"""
function ∂logistic(z::Float64)
	a = exp(z) + 1
    return (1 / a) - (1 / (a ^ 2))
end

"""
    Log PDF of the BernoulliLogit()

A univariate bernoulli logit distribution.
"""
function logpdf_bernoulli_logit(z::Float64, y::Int64)
    return y * z - log1pexp(z)
end

"""
	Log PDF of the Multivariate Normal distribution centred at mean = 0

A mulativariate normal distribution.
"""
function logpdf_centred_mvnormal(σ::Float64, θ::Array{Float64,1})
	return -(length(θ) * (log2π + log(abs2(σ))) + sum(@. abs2(θ / σ))) / 2
	# return -(length(θ) * (Float64(log2π) + log(σ)))/2 - sum(abs2, a.chol.L \ x)/2
end

"""
    PDF of the BernoulliLogit(), z = x * θ

A univariate bernoulli logit distribution.
"""
function pdf_bernoulli_logit(z::Float64, y::Int64)
	y * logistic(z) + (1 - y) * (1 - logistic(z))
end

"""
	Function to calculate v' * A quickly and efficiently
"""
function tt(v, A)
    r = similar(A)
    @inbounds for j = 1:size(A,2)
        @simd for i = 1:size(A,1)
            r[i,j] = v[j] * A[i,j] # fixed a typo here!
        end
    end
    r
end
