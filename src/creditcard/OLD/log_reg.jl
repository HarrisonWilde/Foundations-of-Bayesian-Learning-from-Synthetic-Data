# # Logistic regression

using TransformVariables, LogDensityProblems, DynamicHMC, DynamicHMC.Diagnostics
using MCMCDiagnostics
using Parameters, Statistics, Random, Distributions, StatsFuns
import ForwardDiff              # use for AD

"""
Logistic regression.

For each draw, ``logit(Pr(yᵢ == 1)) ∼ Xᵢ β``. Uses a `β ∼ Normal(0, σ)` prior.

`X` is supposed to include the `1`s for the intercept.
"""
struct BetaLogisticRegression{Ty, TX, Tσ, Tβ, Tβw}
    y::Ty
    X::TX
    σ::Tσ
    β::Tβ
    βw::Tβw
end

function (problem::BetaLogisticRegression)(θ)
    @unpack y_real, X_real, y_synth, X_synth, σ, β, βw = problem
    @unpack γ = θ
    loglik_real = 
    int_term = (1 / (β + 1)) * (exp(logpdf.(Bernoulli.(logistic.(X*γ)), y)) ^ (β + 1) + (1 - exp(logpdf.(Bernoulli.(logistic.(X*γ)), y)) ^ β) ^ (β + 1))
    loglik_synth = βw * sum((1 / β) * exp(logpdf.(Bernoulli.(logistic.(X*γ)), y)) ^ β - int_term)
    logpri = sum(logpdf.(Ref(Normal(0, σ)), γ))
    loglik + logpri
end


# Make up parameters, generate data using random draws.

N = 1000
β = [1.0, 2.0]
X = hcat(ones(N), randn(N))
y = rand.(Bernoulli.(logistic.(X*β)));

# Create a problem, apply a transformation, then use automatic differentiation.

p = LogisticRegression(y, X, 10.0)   # data and (vague) priors
t = as((β = as(Array, length(β)), )) # identity transformation, just to get the dimension
P = TransformedLogDensity(t, p)      # transformed
∇P = ADgradient(:ForwardDiff, P)

# Sample using NUTS, random starting point.

results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000);

# Extract the posterior. (Here the transformation was not really necessary).

β_posterior = first.(transform.(t, results.chain));

# Check that we recover the parameters.

mean(β_posterior)

# Quantiles

qs = [0.05, 0.25, 0.5, 0.75, 0.95]
quantile(first.(β_posterior), qs)

quantile(last.(β_posterior), qs)

# Check that mixing is good.

ess = vec(mapslices(effective_sample_size, reduce(hcat, β_posterior); dims = 2))