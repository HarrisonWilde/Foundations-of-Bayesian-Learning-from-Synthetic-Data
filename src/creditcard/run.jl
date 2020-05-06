using ArgParse
using Distributed
using ForwardDiff
using LinearAlgebra
using CSV
using DataFrames
using AdvancedHMC
using Distributions
using Turing
using Zygote
using Random: seed!
using MCMCChains
using JLD
using MLJ
using Optim
using MLJLinearModels
using Dates
using ProgressMeter
using SharedArrays
using SpecialFunctions
using StatsFuns: log1pexp, log2π
using MLJBase: auc
using SpecialFunctions
include("utils.jl")
include("experiment.jl")
include("distributions.jl")
include("weight_calibration.jl")
include("evaluation.jl")

@everywhere begin
    using ArgParse
    using Distributed
    using ForwardDiff
    using LinearAlgebra
    using CSV
    using DataFrames
    using AdvancedHMC
    using Distributions
    using Turing
    using Zygote
    using Random: seed!
    using MCMCChains
    using JLD
    using MLJ
    using Optim
    using MLJLinearModels
    using Dates
    using ProgressMeter
    using SharedArrays
    using SpecialFunctions
    using StatsFuns: log1pexp, log2π
    using MLJBase: auc
    using SpecialFunctions
    include("src/creditcard/utils.jl")
    include("src/creditcard/experiment.jl")
    include("src/creditcard/distributions.jl")
    include("src/creditcard/weight_calibration.jl")
    include("src/creditcard/evaluation.jl")
    include("src/creditcard/plotting.jl")
end

"""
Christoph Hedtrich:house_with_garden:  13 hours ago
Hi, doesn't it work if you drop the machine file, but do the following in Julia:
using ClusterManagers
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))

Christoph Hedtrich:house_with_garden:  13 hours ago
also I had issues until I have set the JULIA_DEPOT_PATH

Christoph Hedtrich:house_with_garden:  13 hours ago
my .bashrc has a line (replace ... with your depot path):
export JULIA_DEPOT_PATH="---"

Christoph Hedtrich:house_with_garden:  13 hours ago
For me the issue was that the login node and the worker nodes have a separate file system and I had to manually move the depot path to the worker file system, otherwise nothing worked...
"""

function main()
    args = parse_cl()
    name, label, eps = args["dataset"], args["label"], args["eps"]
    folds, split = args["folds"], args["split"]
    use_ad, distributed = args["use_ad"], args["distributed"]
    # name, label, eps, folds, split, distributed, use_ad = "uci_spambase", "label", "6.0", 5, 1.0, true, false
    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    println("Loading data...")
    labels, real_data, synth_data = load_data(name, label, eps)
    println("Setting up experiment...")
    θ_dim = size(real_data)[2] - 1
    w = 0.5
    βw = 0.5
    βw = 1.15
    σ = 50.0
    λ = 0.0
    real_αs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
    synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    αs = get_conditional_pairs(real_αs, synth_αs)
    num_αs = size(αs)[1]
    total_steps = num_αs * folds
    results = SharedArray{Float64, 2}((total_steps, 10))
    bayes_factors = SharedArray{Float64, 3}((4, 4, total_steps))

    n_samples, n_warmup = 100000, 20000
    show_progress = false

    if distributed
        println("Distributing work...")
        p = Progress(total_steps)
        progress_pmap(1:total_steps, progress=p) do i

            fold = ((i - 1) % folds)
            real_α, synth_α = αs[Int(ceil(i / folds))
            X_real, y_real, X_synth, y_synth, X_test, y_test = fold_α(
                real_data, synth_data, real_α, synth_α,
                fold, folds, labels
            )
            metric, initial_θ, βw_calib = init_run(
                θ_dim, λ, X_real, y_real, X_synth, y_synth, β
            )
            print(βw_calib)

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
            hamiltonian_β, proposal_β, adaptor_β = setup_run(
                ℓπ_β,
                ∂ℓπ∂θ_β,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_β, stats_β = sample(
                hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_β, ll_β, bf_β = evalu(X_test, y_test, samples_β)

            # KLD WEIGHTED
            hamiltonian_weighted, proposal_weighted, adaptor_weighted = setup_run(
                ℓπ_weighted,
                ∂ℓπ∂θ_weighted,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_weighted, stats_weighted = sample(
                hamiltonian_weighted, proposal_weighted, initial_θ, n_samples, adaptor_weighted, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_weighted, ll_weighted, bf_weighted = evalu(X_test, y_test, samples_weighted)

            # KLD NAIVE
            hamiltonian_naive, proposal_naive, adaptor_naive = setup_run(
                ℓπ_naive,
                ∂ℓπ∂θ_naive,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_naive, stats_naive = sample(
                hamiltonian_naive, proposal_naive, initial_θ, n_samples, adaptor_naive, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_naive, ll_naive, bf_naive = evalu(X_test, y_test, samples_naive)

            # KLD NO SYNTHETIC
            hamiltonian_no_synth, proposal_no_synth, adaptor_no_synth = setup_run(
                ℓπ_no_synth,
                ∂ℓπ∂θ_no_synth,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_no_synth, stats_no_synth = sample(
                hamiltonian_no_synth, proposal_no_synth, initial_θ, n_samples, adaptor_no_synth, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_no_synth, ll_no_synth, bf_no_synth = evalu(X_test, y_test, samples_no_synth)

            bf_matrix = create_bayes_factor_matrix([bf_β, bf_weighted, bf_naive, bf_no_synth])
            results[i, :] = [real_α, synth_α, auc_β, auc_weighted, auc_naive, auc_no_synth, ll_β, ll_weighted, ll_naive, ll_no_synth]
            bayes_factors[:, :, i] = bf_matrix
        end
    else
        println("Beginning experiment...")
        @showprogress for i in 1:total_steps

            fold = ((i - 1) % folds)
            real_α, synth_α = αs[Int(ceil(i / folds))
            X_real, y_real, X_synth, y_synth, X_test, y_test = fold_α(
                real_data, synth_data, real_α, synth_α,
                fold, folds, labels
            )
            metric, initial_θ, βw_calib = init_run(
                θ_dim, λ, X_real, y_real, X_synth, y_synth, β
            )
            print(βw_calib)

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
            hamiltonian_β, proposal_β, adaptor_β = setup_run(
                ℓπ_β,
                ∂ℓπ∂θ_β,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_β, stats_β = sample(
                hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_β, ll_β, bf_β = evalu(X_test, y_test, samples_β)

            # KLD WEIGHTED
            hamiltonian_weighted, proposal_weighted, adaptor_weighted = setup_run(
                ℓπ_weighted,
                ∂ℓπ∂θ_weighted,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_weighted, stats_weighted = sample(
                hamiltonian_weighted, proposal_weighted, initial_θ, n_samples, adaptor_weighted, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_weighted, ll_weighted, bf_weighted = evalu(X_test, y_test, samples_weighted)

            # KLD NAIVE
            hamiltonian_naive, proposal_naive, adaptor_naive = setup_run(
                ℓπ_naive,
                ∂ℓπ∂θ_naive,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_naive, stats_naive = sample(
                hamiltonian_naive, proposal_naive, initial_θ, n_samples, adaptor_naive, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_naive, ll_naive, bf_naive = evalu(X_test, y_test, samples_naive)

            # KLD NO SYNTHETIC
            hamiltonian_no_synth, proposal_no_synth, adaptor_no_synth = setup_run(
                ℓπ_no_synth,
                ∂ℓπ∂θ_no_synth,
                metric,
                initial_θ,
                use_ad=use_ad
            )
            samples_no_synth, stats_no_synth = sample(
                hamiltonian_no_synth, proposal_no_synth, initial_θ, n_samples, adaptor_no_synth, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_no_synth, ll_no_synth, bf_no_synth = evalu(X_test, y_test, samples_no_synth)

            bf_matrix = create_bayes_factor_matrix([bf_β, bf_weighted, bf_naive, bf_no_synth])
            results[i, :] = [real_α, synth_α, auc_β, auc_weighted, auc_naive, auc_no_synth, ll_β, ll_weighted, ll_naive, ll_no_synth]
            bayes_factors[:, :, i] = bf_matrix
        end
    end

    # Record the results in csv and JLD objects
    results_df = create_results_df(results)
    CSV.write("src/creditcard/outputs/results___$(t).csv", results_df)
    save("src/creditcard/outputs/bayes_factors___$(t).jld", "data", bayes_factors)
    println("Done")
end

main()