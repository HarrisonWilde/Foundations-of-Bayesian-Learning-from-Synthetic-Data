using Distributed
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
using MLJ
using Optim
using MLJLinearModels
using Dates
using ProgressMeter
using SharedArrays
using SpecialFunctions
using StatsFuns: log1pexp, log2π, logistic
using MLJBase: auc
include("utils.jl")
include("distrib_utils.jl")
include("distributions.jl")
include("plotting.jl")

print(nworkers())
print(workers())
# addprocs(6)

@everywhere begin
    using Pkg; Pkg.activate(".")
    using Distributed
    using ForwardDiff
    using LinearAlgebra
    using CSV
    using DataFrames
    using AdvancedHMC
    using Distributions
    using Turing
    using Optim
    using Zygote
    using StatsPlots
    using Random: seed!
    using MCMCChains
    using MLJ
    using MLJLinearModels
    using Dates
    using ProgressMeter
    using SharedArrays
    using SpecialFunctions
    using StatsFuns: log1pexp, log2π, logistic
    using MLJBase: auc
    include("src/creditcard/utils.jl")
    include("src/creditcard/distrib_utils.jl")
    include("src/creditcard/distributions.jl")
    include("src/creditcard/plotting.jl")
end
# theme(:vibrant)

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

w = 0.5
β = 0.5
βw = 1.15
σ = 50.0
λ = 1.0
# num_chains = 2
real_αs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
αs = get_conditional_pairs(real_αs, synth_αs)
num_αs = size(αs)[1]
results = SharedArray{Float64, 2}((num_αs, 12))
bayes_factors = SharedArray{Float64, 3}((5, 5, num_αs))

n_samples, n_warmup = 100000, 20000
show_progress = true
p = Progress(num_αs)
channel = RemoteChannel(()->Channel{Bool}(num_αs), 1)
print("Starting...")

# real_α = 0.1
# synth_α = 0.1
@sync begin
    # this task prints the progress bar
    @async while take!(channel)
        next!(p)
    end

    @async begin
        # @showprogress 1 for i in 1:num_αs
        @distributed for i in 1:num_αs

            real_α, synth_α = αs[i]
            # Take matrix slices according to αs
            X_real = Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)])
            y_real = Int.(real_train[1:floor(Int32, len_real * real_α), labels[1]])
            X_synth = Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
            y_synth = Int.(synth_train[1:floor(Int32, len_synth * synth_α), labels[1]])

            # Define mass matrix and initial guess at θ
            metric = DiagEuclideanMetric(size(X_real)[2])
            # initial_θ = zeros(size(X_real)[2])
            lr1 = LogisticRegression(λ; fit_intercept = false)
            initial_θ = MLJLinearModels.fit(lr1, X_real, y_real, solver=MLJLinearModels.LBFGS())
            auc_mlj, ll_mlj, bf_mlj = evalu(X_test, y_test, [initial_θ])

            lr2 = LogisticRegression(λ; fit_intercept = false)
            θ_0 = MLJLinearModels.fit(lr2, X_synth, y_synth, solver=MLJLinearModels.LBFGS())
            βw = weight_calib(X_synth, y_synth, β, θ_0)

            # Define log posteriors and gradients of them
            ℓπ_β, ∂ℓπ∂θ_β = (
                ℓπ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth),
                ∂ℓπ∂θ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)
            )
            ℓπ_weighted, ∂ℓπ∂θ_weighted = (
                ℓπ_kld(σ, w, X_real, y_real, X_synth, y_synth),
                ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)
            )
            ℓπ_naive, ∂ℓπ∂θ_naive = (
                ℓπ_kld(σ, 1, X_real, y_real, X_synth, y_synth),
                ∂ℓπ∂θ_kld(σ, 1, X_real, y_real, X_synth, y_synth)
            )
            ℓπ_no_synth, ∂ℓπ∂θ_no_synth = (
                ℓπ_kld(σ, 0, X_real, y_real, X_synth, y_synth),
                ∂ℓπ∂θ_kld(σ, 0, X_real, y_real, X_synth, y_synth)
            )

            # BETA DIVERGENCE
            hamiltonian_β, proposal_β, adaptor_β = setup_run(ℓπ_β, ∂ℓπ∂θ_β, metric, initial_θ)
            samples_β, stats_β = sample(
                hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            # chain_β = Chains(samples_β)
            auc_β, ll_β, bf_β = evalu(X_test, y_test, samples_β)

            # KLD WEIGHTED
            hamiltonian_weighted, proposal_weighted, adaptor_weighted = setup_run(
                ℓπ_weighted,
                ∂ℓπ∂θ_weighted,
                metric,
                initial_θ
            )
            samples_weighted, stats_weighted = sample(
                hamiltonian_weighted, proposal_weighted, initial_θ, n_samples, adaptor_weighted, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            # chain_weighted = Chains(samples_weighted)
            auc_weighted, ll_weighted, bf_weighted = evalu(X_test, y_test, samples_weighted)

            # KLD NAIVE
            hamiltonian_naive, proposal_naive, adaptor_naive = setup_run(
                ℓπ_naive,
                ∂ℓπ∂θ_naive,
                metric,
                initial_θ
            )
            samples_naive, stats_naive = sample(
                hamiltonian_naive, proposal_naive, initial_θ, n_samples, adaptor_naive, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            # chain_naive = Chains(samples_naive)
            auc_naive, ll_naive, bf_naive = evalu(X_test, y_test, samples_naive)

            # KLD NO SYNTHETIC
            hamiltonian_no_synth, proposal_no_synth, adaptor_no_synth = setup_run(
                ℓπ_no_synth,
                ∂ℓπ∂θ_no_synth,
                metric,
                initial_θ
            )
            samples_no_synth, stats_no_synth = sample(
                hamiltonian_no_synth, proposal_no_synth, initial_θ, n_samples, adaptor_no_synth, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            # chain_no_synth = Chains(samples_no_synth)
            auc_no_synth, ll_no_synth, bf_no_synth = evalu(X_test, y_test, samples_no_synth)

            bf_matrix = create_bayes_factor_matrix([bf_mlj, bf_β, bf_weighted, bf_naive, bf_no_synth])
            bayes_factors[:, :, i] = bf_matrix
            results[i, :] = [real_α, synth_α, auc_mlj, auc_β, auc_weighted, auc_naive, auc_no_synth, ll_mlj, ll_β, ll_weighted, ll_naive, ll_no_synth]
            put!(channel, true)
            
        end
        put!(channel, false)
    end
end


results_df = create_results_df(results)
CSV.write("src/creditcard/outputs/results___$(t).csv", results_df)
# save("src/creditcard/outputs/bayes_factors___$(t).jld", "data", bayes_factors)


t = "17_48_15__04_05_2020"
results = CSV.read("src/creditcard/outputs/results___$(t).csv", copycols=true)
# bayes_factors = load("src/creditcard/outputs/bayes_factors___$(t).jld")["data"]
sort!(results, (:real_α, :synth_α))
real_αs = unique(results[!, :real_α])
synth_αs = unique(results[!, :synth_α])
plot_all(results, real_αs, synth_αs, ["beta" "weighted" "naive" "no_synth"], ["auc", "ll"], t)



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
