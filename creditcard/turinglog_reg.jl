using Turing, Distributions, MCMCChains, StatsPlots, DynamicHMC
using DataFrames, LinearAlgebra, CSV, Parameters


function get_conditional_pairs(l1, l2; max_sum=1)
	return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
end

struct Data{Ty_real, TX_real, Ty_synth, TX_synth, Tnum_params}
    y_real::Ty_real
    X_real::TX_real
    y_synth::Ty_synth
    X_synth::TX_synth
    num_params::Tnum_params
end

struct KLDParams{Tσ}
	σ::Tσ
end

struct WeightedKLDParams{Tw, Tσ}
    w::Tw
    σ::Tσ
end

struct BetaParams{Tβ, Tβw, Tσ}
    β::Tβ
    βw::Tβw
    σ::Tσ
end

@model weighted_logistic_regression(data, params) = begin
	@unpack y_real, X_real, y_synth, X_synth, num_params = data
	@unpack w, σ = params

    coefs ~ MvNormal(zeros(num_params), Diagonal(repeat([σ], num_params)))
    @logpdf() += sum(logpdf.(BinomialLogit.(1, X_real * coefs), y_real))
    @logpdf() += w * sum(logpdf.(BinomialLogit.(1, X_synth * coefs), y_synth))
end

# function (data::Data, params::WeightedKLDParams)(θ)
#     @unpack y_real, X_real, y_synth, X_synth = data
#     @unpack w, σ = params
#     @unpack γ = θ
#     loglik_real = sum(logpdf.(Bernoulli.(logistic.(X_real * γ)), y_real))
#     loglik_synth = w * sum(logpdf.(Bernoulli.(logistic.(X_real * γ)), y_real))
#     logpri = sum(logpdf.(Ref(Normal(0, σ)), γ))
#     loglik_real + loglik_synth + logpri
# end

# # function (data::Data, params::BetaParams)(θ)
# #     @unpack y_real, X_real, y_synth, X_synth = data
# #     @unpack w, β, βw, σ = params
# #     @unpack γ = θ
# #     loglik_real = sum(logpdf.(Bernoulli.(logistic.(X_real * γ)), y_real))
# #     int_term = (1 / (β + 1)) * (exp(logpdf.(Bernoulli.(logistic.(X_synth * γ)), y_synth)) ^ (β + 1) + (1 - exp(logpdf.(Bernoulli.(logistic.(X_synth * γ)), y_synth)) ^ β) ^ (β + 1))
# #     loglik_synth = βw * sum((1 / β) * exp(logpdf.(Bernoulli.(logistic.(X_synth * γ)), y_synth)) ^ β - int_term)
# #     logpri = sum(logpdf.(Ref(Normal(0, σ)), γ))
# #     loglik + logpri
# # end

# # Make up parameters, generate data using random draws.

# # Create a problem, apply a transformation, then use automatic differentiation.

# p = (
# 	data(y, X, 10.0)
# 	)   # data and (vague) priors
# t = as((γ = as(Array, length(γ)), )) # identity transformation, just to get the dimension
# P = TransformedLogDensity(t, p)      # transformed
# ∇P = ADgradient(:ForwardDiff, P)

# # Sample using NUTS, random starting point.

# results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000);

labels = [Symbol("Class"),]

real_train = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_real_train.csv")
real_train = hcat(ones(size(real_train)[1]), real_train)
real_test = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_real_test.csv")
synth_train = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_synth_train.csv")
synth_train = hcat(ones(size(synth_train)[1]), synth_train)
synth_test = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_synth_test.csv")

len_real = size(real_train)[1]
len_synth = size(synth_train)[1]
len_test = size(real_test)[1]

num_chains = 3
real_αs = [0.1]
synth_αs = [0.0, 0.1, 0.2]
real_α = 0.1
synth_α = 0.1
for (real_α, synth_α) in get_conditional_pairs(real_αs, synth_αs)
	input_data = Data(
		Int.(Matrix(real_train[1:floor(Int32, len_real * real_α), labels])), 
		Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)]), 
		Int.(Matrix(synth_train[1:floor(Int32, len_synth * synth_α), labels])), 
		Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)]),
		size(real_train)[2] - size(labels)[1]
	)
	weighted_params = WeightedKLDParams(0.5, 10.0)
	weighted_chain = mapreduce(c -> sample(weighted_logistic_regression(input_data, weighted_synth_params), NUTS(10, 0.651), 50), chainscat, 1:num_chains)
	write("weighted_chains_real" * string(real_α) * "_synth" * string(synth_α), weighted_synth_chain)
	kld_params = WeightedKLDParams(1.0, 10.0)
	weighted_chain = mapreduce(c -> sample(weighted_logistic_regression(input_data, weighted_synth_params), NUTS(10, 0.651), 50), chainscat, 1:num_chains)
	# weighted_synth_chain = mapreduce(c -> sample(weighted_logistic_regression(input_data, weighted_synth_params), DynamicNUTS(), 2500), chainscat, 1:num_chains)
end
