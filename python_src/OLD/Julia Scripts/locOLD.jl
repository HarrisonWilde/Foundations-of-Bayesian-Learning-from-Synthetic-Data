using Turing, StatsPlots, Bijectors, Random, Plots, Distributions

@model location_variance(n) = begin
	# Assumptions
	τ ~ Gamma(2,2)
	μ ~ Normal(0, n * τ)

	# Observations
	for i in 1:n
		x = rand(Normal(μ, τ)) + rand(Laplace())
		x ~ Normal(μ, τ)
	end
	return μ, τ
end

@model loc(xs) = begin
	σ ~ Gamma(2,2)
	μ ~ Normal(0, σ)
	for x in xs
		x ~ Normal(μ, σ)
	end
	return σ, μ
end

μ = 0
σ = 2
n = 20
x = rand(Normal(μ, σ), n) + rand(Laplace(0,2), n)

chn = sample(loc(μ, σ, x), HMC(0.1, 5), 1000)
StatsPlots.plot(chn)

@model loc_var(μ, σ, n) = begin
	#begin block body
end

x = [1.5, 2.0, 13.0, 2.1, 0.0]

# Set up the model call, sample from the prior.
model = location_variance(1000)
vi = Turing.VarInfo()
model(vi, Turing.SampleFromPrior())
vi.flags["trans"] = [true, false]

function sim_contam_data(n, μc, σc, μ, σ, ϵ)
	x = rand()
end

# Evaluate surface at coordinates.
function evaluate(m1, m2)
    vi.vals .= [m1, m2]
    model(vi, Turing.SampleFromPrior())
    -vi.logp
end

function plot_sampler(chain)
    # Extract values from chain.
    val = get(chain, [:τ, :μ, :lp])
    ss = link.(Ref(InverseGamma(2, 3)), val.τ)
    ms = val.μ
    lps = val.lp

    # How many surface points to sample.
    granularity = 500

    # Range start/stop points.
    spread = 0.5
    σ_start = minimum(ss) - spread * std(ss); σ_stop = maximum(ss) + spread * std(ss);
    μ_start = minimum(ms) - spread * std(ms); μ_stop = maximum(ms) + spread * std(ms);
    σ_rng = collect(range(σ_start, stop=σ_stop, length=granularity))
    μ_rng = collect(range(μ_start, stop=μ_stop, length=granularity))

    # Make surface plot.
    p = surface(σ_rng, μ_rng, evaluate,
          camera=(30, 65),
          ticks=nothing,
          colorbar=false,
          color=:inferno)

    line_range = 1:length(ms)

    plot3d!(ss[line_range], ms[line_range], -lps[line_range],
        lc =:viridis, line_z=collect(line_range),
        legend=false, colorbar=false, alpha=0.5)

    return p
end

c = sample(model, MH(), 1000)
plot_sampler(c)