using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)

# Summarise results (currently requires the master branch from MCMCChains)
describe(chn)

# Plot and save results
p = plot(chn)
savefig("gdemo-plot.png")

