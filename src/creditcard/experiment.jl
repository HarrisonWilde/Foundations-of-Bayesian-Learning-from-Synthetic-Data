using ForwardDiff
using LinearAlgebra
using CSV
using DataFrames
using AdvancedHMC
using Distributions
using Turing
using Zygote
using StatsPlots
using Random: seed!
using MCMCChains
using MLJLinearModels
using Dates
includet("utils.jl")
includet("distrib_utils.jl")
includet("distributions.jl")
includet("plotting.jl")
theme(:vibrant)

t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
results_name = "results___$(t)"

# Load in data generated and split into train test by PATE-GAN

# labels = [Symbol("Class"),]
# real_train = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_real_train.csv")
# real_test = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_real_test.csv")
# synth_train = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_synth_train.csv")
# synth_test = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_synth_test.csv")

labels = [Symbol("DefaultNum"),]
real_train = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_real_train.csv")
real_test = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_real_test.csv")
synth_train = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_synth_train.csv")
synth_test = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_synth_test.csv")

# Append 1s column to each to allow for intercept term in lin regression
real_train = hcat(ones(size(real_train)[1]), real_train)
real_test = hcat(ones(size(real_test)[1]), real_test)
synth_train = hcat(ones(size(synth_train)[1]), synth_train)
synth_test = hcat(ones(size(synth_test)[1]), synth_test)

len_real = size(real_train)[1]
len_synth = size(synth_train)[1]
len_test = size(real_test)[1]

X_test = Matrix(real_test[:, Not(labels)])
y_test = Int.(real_test[:, labels[1]])

results = result_storage()

w = 0.5
β = 0.5
βw = 1.15
σ = 50.0
λ = 1.0
num_chains = 2
real_αs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
synth_αs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]
αs = get_conditional_pairs(real_αs, synth_αs)
n_samples, n_warmup = 100000, 10000
show_progress = false
# real_α = 0.0
# synth_α = 0.0

for (real_α, synth_α) in αs

    # Take matrix slices according to αs
    X_real = Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)])
    y_real = Int.(real_train[1:floor(Int32, len_real * real_α), labels[1]])
    X_synth = Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
    y_synth = Int.(synth_train[1:floor(Int32, len_synth * synth_α), labels[1]])

    # Define log posteriors and gradients of them
    ℓπ_weighted, ∂ℓπ∂θ_weighted = (
        ℓπ_kld(σ, w, X_real, y_real, X_synth, y_synth),
        ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)
    )
    ℓπ_β, ∂ℓπ∂θ_β = (
        ℓπ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth),
        ∂ℓπ∂θ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)
    )
    ℓπ_naive, ∂ℓπ∂θ_naive = (
        ℓπ_kld(σ, 1, X_real, y_real, X_synth, y_synth),
        ∂ℓπ∂θ_kld(σ, 1, X_real, y_real, X_synth, y_synth)
    )
    ℓπ_no_synth, ∂ℓπ∂θ_no_synth = (
        ℓπ_kld(σ, 0, X_real, y_real, X_synth, y_synth),
        ∂ℓπ∂θ_kld(σ, 0, X_real, y_real, X_synth, y_synth)
    )

    # Define mass matrix and initial guess at θ
    metric = DiagEuclideanMetric(size(X_real)[2])
    # initial_θ = zeros(size(X_real)[2])
    lr = LogisticRegression(λ; fit_intercept = false)
    initial_θ = MLJLinearModels.fit(lr, X_real, y_real, solver=LBFGS())
    auc_mlj = evalu(X_test, y_test, [initial_θ])

    # BETA DIVERGENCE
    hamiltonian_β, proposal_β, adaptor_β = setup_run(ℓπ_β, ∂ℓπ∂θ_β, metric, initial_θ)
    samples_β, stats_β = sample(
        hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
        drop_warmup=true, progress=show_progress
    )
    # chain_β = Chains(samples_β)
    auc_β = evalu(X_test, y_test, samples_β)

    # KLD WEIGHTED
    hamiltonian_weighted, proposal_weighted, adaptor_weighted = setup_run(
        ℓπ_weighted,
        ∂ℓπ∂θ_weighted,
        metric,
        initial_θ
    )
    samples_weighted, stats_weighted = sample(
        hamiltonian_weighted, proposal_weighted, initial_θ, n_samples, adaptor_weighted, n_warmup;
        drop_warmup=true, progress=show_progress
    )
    # chain_weighted = Chains(samples_weighted)
    auc_weighted = evalu(X_test, y_test, samples_weighted)

    # KLD NAIVE
    hamiltonian_naive, proposal_naive, adaptor_naive = setup_run(
        ℓπ_naive,
        ∂ℓπ∂θ_naive,
        metric,
        initial_θ
    )
    samples_naive, stats_naive = sample(
        hamiltonian_naive, proposal_naive, initial_θ, n_samples, adaptor_naive, n_warmup;
        drop_warmup=true, progress=show_progress
    )
    # chain_naive = Chains(samples_naive)
    auc_naive = evalu(X_test, y_test, samples_naive)

    # KLD NO SYNTHETIC
    hamiltonian_no_synth, proposal_no_synth, adaptor_no_synth = setup_run(
        ℓπ_no_synth,
        ∂ℓπ∂θ_no_synth,
        metric,
        initial_θ
    )
    samples_no_synth, stats_no_synth = sample(
        hamiltonian_no_synth, proposal_no_synth, initial_θ, n_samples, adaptor_no_synth, n_warmup;
        drop_warmup=true, progress=show_progress
    )
    # chain_no_synth = Chains(samples_no_synth)
    auc_no_synth = evalu(X_test, y_test, samples_no_synth)

    push!(results, (real_α, synth_α, auc_mlj, auc_β, auc_weighted, auc_naive, auc_no_synth))
    CSV.write("src/creditcard/outputs/$(results_name).csv", results)

end

plot_all(results, real_αs, t)


# function evaluate(X_test::Matrix, y_test::Array, chain, threshold)
#     # Pull the means from each parameter's sampled values in the chain.
#
#     # num_iters, _ = size(Array(chn))
#
#     # log_likes = Vector{Array{Float64}}(undef, num_iters)
#
#     # @showprogress for i in 1:num_iters
#     #     log_likes[i] = logpdf.(BinomialLogit.(1, X_test * Array(chn)[i, :]), y_test)
#     # end
#
#     # log_loss = -mean(logsumexp.(log_likes) .- log(length(Array(chn)[:, 1])))
#
#     mean_coefs = [mean(chain[Symbol("coef$i")].value) for i in 1:31]
#
#     # Retrieve the number of rows.
#     n, _ = size(X_test)
#
#     # Generate a vector to store our predictions.
#     preds = Vector{Float64}(undef, n)
#     probs = Vector{Float64}(undef, n)
#
#     # Calculate the logistic function for each element in the test set.
#     for i in 1:n
#         prob = logistic(dot(X_test[i, :], mean_coefs))
#         probs[i] = prob
#         if prob >= threshold
#             preds[i] = 1
#         else
#             preds[i] = 0
#         end
#     end
#     return(probs, preds)
# end
#
#

# real_α = 0.25
# synth_α = 0.25
# # for (real_α, synth_α) in αs
#     # chn = read("toy_weighted_chains_real$(real_α)_synth$(synth_α)", Chains)
#     # print(chn)
#     probs, preds = evaluate(X_test, y_test, samples, 0.5)
#     r = ROCAnalysis.roc(probs, y_test)
#     print("\nReal: $(real_α), Synthetic: $(synth_α), AUC: $(auc(r))")
# end
