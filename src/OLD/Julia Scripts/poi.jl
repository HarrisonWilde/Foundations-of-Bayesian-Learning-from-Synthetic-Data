using Gadfly, RDatasets, Turing, Distributions, StatsPlots, Divergences

grouseticks = dataset("lme4", "grouseticks")

describe(grouseticks)

y = grouseticks[:, :Ticks]

@model simple_poisson(y, α, β) = begin
	λ ~ InverseGamma(α, β)
	for i in eachindex(y)
		y[i] ~ Poisson(λ)
	end
end

n = 1500

#sample from the chain
model1 = simple_poisson(y, 2, 2)
chn1 = sample(model1, NUTS(0.65), n)
StatsPlots.histogram(rand(Poisson(chn1["λ"].value[500]), 10000))



@model zi_poisson(y, α, β, ρ̂) = begin

	λ1 ~ InverseGamma(α, β)
	λ2 ~ Poisson(0)
	λ = [λ1, λ2]

	ρ = rand(Uniform(0, ρ̂))
	w = [ρ, 1 - ρ]

	k = Vector{Int}(undef, length(y))

	for i in eachindex(y)
		k[i] ~ Categorical(w)
		y[i] ~ Poisson(λ[k[i]])
	end
	return k
end

n = 200

#sample from the chain
model2 = zi_poisson(y, 2, 2, 0.2)
sampler2 = Gibbs(PG(100, :k), HMC(0.05, 10, :λ1, :λ2))
chn2 = sample(model2, sampler2, n)
StatsPlots.histogram(rand(Poisson(chn2["λ"].value[500]), 10000))



@model mix_poisson(x, y, ρ̂) = begin

	λ1 ~ InverseGamma(α, β)
	λ2 ~ InverseGamma(α, β)
	λ = [λ1, λ2]

	ρ ~ Uniform(0, ρ̂)
	w = [ρ, 1 - ρ]
	k = Vector{Int}(undef, length(y))

	for i in eachindex(y)
		k[i] ~ Categorical(w)
		y[i] ~ Poisson(λ[k[i]])
	end
end

n = 200

#sample from the chain
model3 = mix_poisson(y, 2, 2, 0.2)
sampler3 = Gibbs(PG(100, :k), HMC(0.05, 10, :λ1, :λ2, :ρ))
chn3 = sample(model3, sampler3, n)
StatsPlots.histogram(rand(Poisson(chn3["λ"].value[500]), 10000))