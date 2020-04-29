using ForwardDiff
using LinearAlgebra
using CSV
using DataFrames
using AdvancedHMC
using Distributions
using Turing
includet("utils.jl")
includet("densities.jl")

labels = [Symbol("Class"),]

# Load in data generated and split into train test by PATE-GAN
# Append 1s column to each to allow for intercept term in lin regression
real_train = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_real_train.csv")
real_train = hcat(ones(size(real_train)[1]), real_train)
real_test = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_real_test.csv")
real_test = hcat(ones(size(real_test)[1]), real_test)
synth_train = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_synth_train.csv")
synth_train = hcat(ones(size(synth_train)[1]), synth_train)
synth_test = CSV.read("data/splits/creditcard_Class_split0.5_ganpate_eps6.0_synth_test.csv")
synth_test = hcat(ones(size(synth_test)[1]), synth_test)

len_real = size(real_train)[1]
len_synth = size(synth_train)[1]
len_test = size(real_test)[1]


w = 0.5
β = 0.5
βw = 1.1
σ = 10.0
num_chains = 2
# real_αs = [0.1, 0.25, 0.5, 1.0]
# synth_αs = [0.05, 0.1, 0.25, 0.5]
# αs = get_conditional_pairs(real_αs, synth_αs)
n_samples, n_warmup = 5000, 1000
real_α = 0.2
synth_α = 0.2
# for (real_α, synth_α) in αs

# Take slices
X_real = Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)])
y_real = Int.(Matrix(real_train[1:floor(Int32, len_real * real_α), labels]))
X_synth = Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
y_synth = Int.(Matrix(synth_train[1:floor(Int32, len_synth * synth_α), labels]))

# Define log posts and gradient functions of θ, opt calculates the same thing but should be faster (uses @. macro grouped broadcasting) but keeping both for now to run comparisons
ℓπ_weighted, ∂ℓπ∂θ_weighted = ℓπ_kld(σ, w, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)
ℓπ_β, ∂ℓπ∂θ_β = ℓπ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)
ℓπ_weighted_opt, ∂ℓπ∂θ_weighted_opt = ℓπ_kld_opt(σ, w, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_kld_opt(σ, w, X_real, y_real, X_synth, y_synth)
ℓπ_β_opt, ∂ℓπ∂θ_β_opt = ℓπ_beta_opt(σ, β, βw, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_beta_opt(σ, β, βw, X_real, y_real, X_synth, y_synth)
# ℓπ_naive, ∂ℓπ∂θ_naive = ℓπ_kld(σ, 1, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_kld(σ, 1, X_real, y_real, X_synth, y_synth)
# ℓπ_no_synth, ∂ℓπ∂θ_no_synth = ℓπ_kld(σ, 0, X_real, y_real, X_synth, y_synth), ∂ℓπ∂θ_kld(σ, 0, X_real, y_real, X_synth, y_synth)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(size(X_real)[2])
hamiltonian = Hamiltonian(
    metric,
    ℓπ_weighted,
    ∂ℓπ∂θ_weighted,
)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_θ = randn(size(X_real)[2])
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.7, integrator))

# Run the sampler
samples, stats = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_warmup; progress=true)
write("src/creditcard/samples_and_stats/weighted_chain_real" * string(real_α) * "_synth" * string(synth_α), samples)
write("src/creditcard/samples_and_stats/weighted_stats_real" * string(real_α) * "_synth" * string(synth_α), stats)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(size(X_real)[2])
hamiltonian = Hamiltonian(
    metric,
    ℓπ_β,
    ∂ℓπ∂θ_β,
)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_θ = randn(size(X_real)[2])
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler
samples2, stats2 = sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_warmup; progress=true)
write("src/creditcard/samples_and_stats/beta_chain_real" * string(real_α) * "_synth" * string(synth_α), samples2)
write("src/creditcard/samples_and_stats/beta_stats_real" * string(real_α) * "_synth" * string(synth_α), stats2)

# end


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


# X_test = Matrix(real_test[:, Not(labels)])
# y_test = Array{Float64}(real_test[:, labels[1]])
# real_α = 0.1
# synth_α = 0.1
# for (real_α, synth_α) in αs
#     chn = read("toy_weighted_chains_real$(real_α)_synth$(synth_α)", Chains)
#     print(chn)
#     probs, preds = evaluate(X_test, y_test, chn, 0.5)
#     r = ROCAnalysis.roc(probs, y_test)
#     print("\nReal: $(real_α), Synthetic: $(synth_α), AUC: $(auc(r))")
# end