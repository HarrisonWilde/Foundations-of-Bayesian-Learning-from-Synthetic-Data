using Mamba, LinearAlgebra, Distributions, Distances

## Model Specification

model = Model(

  x = Stochastic(1,
    (μ, σ2) ->  Normal(μ, sqrt(σ2)),
    false
  ),

  μ = Stochastic(
    (σ2) -> Normal(μ, sqrt(σ2)),
  ),

  σ2 = Stochastic(
    () -> InverseGamma(0.1, 0.1),
  )

)

scheme = [NUTS([:μ, :σ2])]

setsamplers!(model, scheme)

data = Dict{Symbol, Array}(
  :x => rand(Normal(2,2), 500)
)

inits = [
  Dict{Symbol, Any}(
    :x => data[:x],
    :σ2 => rand(InverseGamma(2,2)),
    :μ => rand(Normal(0, rand(InverseGamma(2,2))))
  ) for i in 1:5
]

sim = mcmc(model, data, inits, 100000, burnin=250, thin=2, chains=5)

draw(plot(sim), filename="summaryplot.svg")
draw(plot(sim, [:autocor, :mean], legend=true), nrow=3, ncol=2, filename="autocormeanplot.svg")
