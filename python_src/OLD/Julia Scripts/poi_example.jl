using Gadfly
using RDatasets, Turing, Distributions


grouseticks = dataset("lme4", "grouseticks")
# bioChemists = dataset("pscl", "bioChemists")

describe(grouseticks)
# describe(bioChemists)

fig1 = plot(grouseticks, x=:Ticks, Geom.histogram)
fig2 = plot(grouseticks, x=:Ticks, Geom.density)

grouseticks = select(grouseticks, Not(:Index))

grouseticks.Year = parse.(Int8, grouseticks.Year)

data = Matrix(select(grouseticks, Not(:Ticks)))
labels = grouseticks[:, :Ticks]

function ℓKLD(x, likelihood)
	-log(likelihood(x))
end

function ℓβ(x, β, likelihood)
	- (likelihood(x)) / (β - 1) 

function ℓTVD(x, likelihood, gn)
	0.5 * abs(1 - likelihood(x) / gn(x))
end

function poi_likelihood(λ, x)
	(exp(-length(x) * λ) * λ^(sum(x))) / prod(map(item -> factorial(big(item)), x))
end

function gamma_prior(λ, α, β)
	((β ^ α) * λ ^ (α - 1) * exp(-β * λ)) / gamma(2)
end

function poi_gamma_posterior(λ, α, β, x, ℓ)
	quadgk(gamma_prior(λ) * exp(-sum(map(item -> ℓ(item, poi_likelihood(λ, item))))), )

post_pred = quadgk(poi_likelihood(λ, labels) * 1)


# Generate your own data
# Look at source code for HMC to figure out where it puts out the density
# Try and do break points 

@model simple_poisson(y, α, β) = begin
	λ ~ Gamma(α, β)
	n = length(y)
	for i in 1:n
		λ ~ Gamma(y[i] + α, length(y) + β)
		x ~ Poisson(λ)
	end
end

chain = mapreduce(c -> sample(simple_poisson(labels, 2, 2), NUTS(200, 0.65), 2500, discard_adapt=false), chainscat, 1:num_chains)

# Visualise the posterior by plotting it
Plots.plot(chain)



@model zi_poisson(x, y, ρ̂) = begin
	λ ~ Gamma(2, 2)
	ρ ~ Uniform(0, ρ̂)
	for i = 1:n
		y[i] ~ (1-ρ) Poisson(λ) + ρ * Zeros(y[i])
	end
end

@model mix_poisson(x, y, ρ̂) = begin
	λ ~ Gamma(2, 2)
	λc ~ Gamma(2, 2)
	ρ ~ Uniform(0, ρ̂)
	for i = 1:n
		y[i] ~ (1-ρ) Poisson(λ) + ρ * Poisson(λc)
	end
end


simplePoi ~ Poisson(λ)
# bcZeroIPoi ~ (1 - ρbc)Poisson(λ) + ρbc
gtZeroIPoi ~ (1 - ρgt) * Poisson(λ) + ρgt
# bcPoiMix ~ (1 - ρbc)Poisson(λ) + ρbc Poisson(λc)
gtPoiMix ~ (1 - ρgt) * Poisson(λ) + ρgt * Poisson(λc)

