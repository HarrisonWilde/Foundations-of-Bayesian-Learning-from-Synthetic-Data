# Import the package.
using AdvancedMH
using Distributions

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(0, 2), 500)

# Define the components of a basic model.
insupport(σ2) = σ2 >= 0
dist(μ, σ2) = Normal(μ, σ2)
prior(σ2) = InverseGamma(2, 2)
σ2 = rand(InverseGamma(2, 2))
prior(μ) = Normal(0, σ2)
μ = rand(Normal(0, σ2))


density(μ, σ2) = insupport(σ2) ? sum(logpdf.(dist(μ, σ2), data)) + logpdf(prior(μ), μ) + logpdf(prior(σ2), σ2) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000; param_names=["μ", "σ2"])




# Import the package.
using AdvancedMH
using Distributions

# Generate a set of data from the posterior we want to estimate.
data = rand(Normal(0, 1), 30)

# Define the components of a basic model.
insupport(θ) = θ[2] >= 0
dist(θ) = Normal(θ[1], θ[2])
density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

# Construct a DensityModel.
model = DensityModel(density)

# Set up our sampler with initial parameters.
spl = MetropolisHastings([0.0, 0.0])

# Sample from the posterior.
chain = sample(model, spl, 100000; param_names=["μ", "σ"])