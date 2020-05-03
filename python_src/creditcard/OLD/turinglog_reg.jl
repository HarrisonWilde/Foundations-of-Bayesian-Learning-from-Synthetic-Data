using Turing, Distributions, MCMCChains, StatsPlots, DynamicHMC, AdvancedMH, StatsFuns, ReverseDiff, Zygote, Tracker
using DataFrames, LinearAlgebra, CSV, Parameters, ProgressMeter


function get_conditional_pairs(l1, l2; max_sum=1)
    return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
end

struct Data{Ty_real, TX_real, Ty_synth, TX_synth}
    y_real::Ty_real
    X_real::TX_real
    y_synth::Ty_synth
    X_synth::TX_synth
end

struct KLDParams{Tμs, Tσs}
    μs::Tμs
    σs::Tσs
end

struct WeightedKLDParams{Tw, Tμs, Tσs}
    w::Tw
    μs::Tμs    
    σs::Tσs
end

struct BetaParams{Tβ, Tβw, Tσ}
    β::Tβ
    βw::Tβw
    σ::Tσ
end

# Zygote, Tracker
# code_warntype
@model weighted_logistic_regression(data, params) = begin

    @unpack y_real, X_real, y_synth, X_synth = data
    @unpack w, μs, σs = params

    coefs ~ MvNormal(μs, σs)
    @logpdf() += sum(logpdf.(BinomialLogit.(1, X_real * coefs), y_real))
    @logpdf() += w * sum(logpdf.(BinomialLogit.(1, X_synth * coefs), y_synth))
end


function weighted_log_reg(data::Data, params::WeightedKLDParams)
    @unpack y_real, X_real, y_synth, X_synth = data
    @unpack w, μs, σs = params

    function get_density(coefs)
        logpri = sum(logpdf.(Ref(Normal(0, σs[1])), coefs))
        loglik_real = sum(logpdf.(BinomialLogit.(1, X_real * coefs), y_real))
        loglik_synth = w * sum(logpdf.(BinomialLogit.(1, X_synth * coefs), y_synth))
        return (loglik_real + loglik_synth + logpri)
    end

    return get_density
end



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
#   data(y, X, 10.0)
#   )   # data and (vague) priors
# t = as((γ = as(Array, length(γ)), )) # identity transformation, just to get the dimension
# P = TransformedLogDensity(t, p)      # transformed
# ∇P = ADgradient(:ForwardDiff, P)

# # Sample using NUTS, random starting point.

# results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇P, 1000);

labels = [Symbol("Class"),]

real_train = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_real_train.csv")
real_train = hcat(ones(size(real_train)[1]), real_train)
real_test = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_real_test.csv")
real_test = hcat(ones(size(real_test)[1]), real_test)
synth_train = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_synth_train.csv")
synth_train = hcat(ones(size(synth_train)[1]), synth_train)
synth_test = CSV.read("./data/splits/creditcard_Class_split0.6_ganpate_eps1.0_synth_test.csv")
synth_test = hcat(ones(size(synth_test)[1]), synth_test)

# n = 300000
# p = 40
# real_coefs = rand(p) .* 2 .- 1
# real_coefs[10] = real_coefs[10] * 10
# real_data = DataFrame(randn(n, p))
# synth_data = DataFrame(Matrix(real_data) .+ rand(Laplace(0, 0.2), (n, p)))
# logit(x) = 1 / (1 + exp(-x))
# classes = rand.(Bernoulli.(logit.(Array(real_data) * real_coefs)))
# real_data.x0 = ones(n)
# real_data.Class = classes
# synth_data.x0 = ones(n)
# synth_data.Class = classes
# real_train = real_data[1:Int(n/2), :]
# real_test = real_data[Int(n/2)+1:n, :]
# synth_train = synth_data[1:Int(n/2), :]
# real_train = synth_data[Int(n/2)+1:n, :]

len_real = size(real_train)[1]
len_synth = size(synth_train)[1]
len_test = size(real_test)[1]

num_chains = 2
real_αs = [0.1, 0.25, 0.5, 1.0]
synth_αs = [0.05, 0.1, 0.25, 0.5]
αs = get_conditional_pairs(real_αs, synth_αs)

Turing.setadbackend(:zygote)
Turing.setcache(true)
for (real_α, synth_α) in αs
    input_data = Data(
        Int.(Matrix(real_train[1:floor(Int32, len_real * real_α), labels])), 
        Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)]), 
        Int.(Matrix(synth_train[1:floor(Int32, len_synth * synth_α), labels])), 
        Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
    )
    params = WeightedKLDParams(
        0.5,
        ones(size(real_train)[2] - size(labels)[1]),
        Diagonal(repeat([2.0], size(real_train)[2] - size(labels)[1]))
    )
    # weighted_chain = mapreduce(c -> sample(weighted_logistic_regression(input_data, weighted_params), DynamicNUTS(), 5000), chainscat, 1:num_chains)
    # 
    weighted_chain = sample(weighted_logistic_regression(input_data, params), NUTS(500, 0.651), 5000)
    write("weighted_chains_real" * string(real_α) * "_synth" * string(synth_α), weighted_chain)
    params = WeightedKLDParams(
        1.0,
        ones(size(real_train)[2] - size(labels)[1]),
        Diagonal(repeat([2.0], size(real_train)[2] - size(labels)[1]))
    )
    # weighted_chain = mapreduce(c -> sample(weighted_logistic_regression(input_data, kld_params), DynamicNUTS(), 5000), chainscat, 1:num_chains)
    naive_chain = sample(weighted_logistic_regression(input_data, params), NUTS(500, 0.651), 5000)
    write("naive_chains_real" * string(real_α) * "_synth" * string(synth_α), naive_chain)
end


function ℓπ_beta(σ, β, βw, θ, X_real, y_real, X_synth, y_synth)
    ℓprior = sum(logpdf.(MvNormal(4, σ), θ))
    ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    ℓsynth = βw * sum(
        (1 / β) * pdf.(BinomialLogit.(1, X_synth * θ), y_synth) ^ β
        - (1 / (β + 1)) * (
            pdf.(BinomialLogit.(1, X_synth * θ), y_synth) ^ (β + 1)
            + (1 - pdf.(BinomialLogit.(1, X_synth * θ), y_synth)) ^ (β + 1)
        )
    )
    return (ℓprior + ℓreal + ℓsynth)
end


function ℓπ_kld(σ, w, θ, X_real, y_real, X_synth, y_synth)
    ℓprior = sum(logpdf.(MvNormal(4, σ), θ))
    ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    ℓsynth = w * sum(logpdf.(BinomialLogit.(1, X_synth * θ), y_synth))
    return (ℓprior + ℓreal + ℓsynth)
end


function ∂ℓπ∂θ_beta(σ, β, βw, θ, X_real, y_real, X_synth, y_synth)
    ∂ℓprior∂θ = sum(logpdf.(Ref(Normal(0, σ)), θ))
    ∂ℓreal∂θ = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    pmf = exp(sum((θ / (1 + exp.(X_synth * θ))
    ∂ℓsynth∂θ = βw * sum(
        (1 / β) * pdf.(BinomialLogit.(1, X_synth * θ), y_synth) ^ β
        - (1 / (β + 1)) * (
            pdf.(BinomialLogit.(1, X_synth * θ), y_synth) ^ (β + 1)
            + (1 - pdf.(BinomialLogit.(1, X_synth * θ), y_synth)) ^ (β + 1)
        )
    )
end


function ∂ℓπ∂θ_kld(σ, w, θ, X_real, y_real, X_synth, y_synth)
    log_prior = sum(logpdf.(Ref(Normal(0, σ)), θ))
    loglik_real = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    loglik_synth = w * sum(logpdf.(BinomialLogit.(1, X_synth * θ), y_synth))
end


for (real_α, synth_α) in αs
    input_data = Data(
        Int.(Matrix(real_train[1:floor(Int32, len_real * real_α), labels])), 
        Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)]), 
        Int.(Matrix(synth_train[1:floor(Int32, len_synth * synth_α), labels])), 
        Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
    )
    params = WeightedKLDParams(
        0.5,
        ones(size(real_train)[2] - size(labels)[1]),
        Diagonal(repeat([2.0], size(real_train)[2] - size(labels)[1]))
    )
    metric = UnitEuclideanMetric(size(real_train)[2])
    Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
    spl = RWMH(
        # Replace with MAP estimator
        MvNormal(
            ones(size(real_train)[2] - size(labels)[1]), 
            Diagonal(repeat([1.0], size(real_train)[2] - size(labels)[1]))
        )
    )
    @code_warntype weighted_log_reg(input_data, params)     
    weighted_chain = mapreduce(c -> sample(DensityModel(weighted_log_reg(input_data, params)), spl, 100000; param_names=["coef$i" for i=1:31], chain_type=Chains), chainscat, 1:num_chains)
    # print(describe(weighted_chain))
    write("weighted_chains_real" * string(real_α) * "_synth" * string(synth_α), weighted_chain)
end


function evaluate(X_test::Matrix, y_test::Array, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain.

    # num_iters, _ = size(Array(chn))

    # log_likes = Vector{Array{Float64}}(undef, num_iters)

    # @showprogress for i in 1:num_iters
    #     log_likes[i] = logpdf.(BinomialLogit.(1, X_test * Array(chn)[i, :]), y_test)
    # end

    # log_loss = -mean(logsumexp.(log_likes) .- log(length(Array(chn)[:, 1])))

    mean_coefs = [mean(chain[Symbol("coef$i")].value) for i in 1:31]

    # Retrieve the number of rows.
    n, _ = size(X_test)

    # Generate a vector to store our predictions.
    preds = Vector{Float64}(undef, n)
    probs = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        prob = logistic(dot(X_test[i, :], mean_coefs))
        probs[i] = prob
        if prob >= threshold
            preds[i] = 1
        else
            preds[i] = 0
        end
    end
    return(probs, preds)
end


X_test = Matrix(real_test[:, Not(labels)])
y_test = Array{Float64}(real_test[:, labels[1]])
real_α = 0.1
synth_α = 0.1
for (real_α, synth_α) in αs
    chn = read("toy_weighted_chains_real$(real_α)_synth$(synth_α)", Chains)
    print(chn)
    probs, preds = evaluate(X_test, y_test, chn, 0.5)
    r = ROCAnalysis.roc(probs, y_test)
    print("\nReal: $(real_α), Synthetic: $(synth_α), AUC: $(auc(r))")
end
