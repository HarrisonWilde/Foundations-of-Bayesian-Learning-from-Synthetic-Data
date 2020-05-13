using ClusterManagers
using Distributed
# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
println("We are all connected and ready.")
for i in workers()
    host, pid = fetch(@spawnat i (gethostname(), getpid()))
    println(host, pid)
end
using ArgParse
using ForwardDiff
using LinearAlgebra
using CSV
using DataFrames
using AdvancedHMC
using Distributions
using Turing
using Zygote
using MCMCChains
using JLD
using MLJ
using Optim
using MLJLinearModels
using Dates
using ProgressMeter
using SharedArrays
using SpecialFunctions
using Random: seed!
using StatsFuns: log1pexp, log2π
using MLJBase: auc
include("utils.jl")
include("experiment.jl")
include("mathematical_utils.jl")
include("distributions.jl")
include("weight_calibration.jl")
include("evaluation.jl")

@everywhere begin
    using Distributed
    using ArgParse
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
    include("src/logistic_regression/utils.jl")
    include("src/logistic_regression/experiment.jl")
    include("src/logistic_regression/mathematical_utils.jl")
    include("src/logistic_regression/distributions.jl")
    include("src/logistic_regression/weight_calibration.jl")
    include("src/logistic_regression/evaluation.jl")
end


function main()
    args = parse_cl()
    path, name, label, ε = args["path"], args["dataset"], args["label"], args["epsilon"]
    iterations, folds, split = args["iterations"], args["folds"], args["split"]
    use_ad, distributed, sampler = args["use_ad"], args["distributed"], args["sampler"]
    # path, name, label, ε, iterations, folds, split, distributed, use_ad, sampler = "src", "uci_heart", "target", "6.0", 1, 5, 1.0, false, false, "Stan"
    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")

    labels, real_data, synth_data = load_data(name, label, ε)
    println("Setting up experiment...")
    θ_dim = size(real_data)[2] - 1
    w = 0.5
    β = 0.5
    βw = 1.15
    σ = 50.0
    λ = 0.0
    real_αs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
    synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    αs = get_conditional_pairs(real_αs, synth_αs)
    num_αs = size(αs)[1]
    total_steps = num_αs * folds
    # results = SharedArray{Float64, 2}("results", (total_steps, 10))
    # bayes_factors = SharedArray{Float64, 3}("bayes_factors", (4, 4, total_steps))
    n_samples, n_warmup = 10000, 2000
    n_chains = 1
    show_progress = true

    if distributed

        p = Progress(total_steps)
        io = open("$(path)/$(name)_$(t)_out.csv", "w")
        write(io, "iter,fold,real_α,synth_α,beta_auc,weighted_auc,naive_auc,no_synth_auc,beta_ll,weighted_ll,naive_ll,no_synth_ll\n")
        close(io)

        if sampler == "Stan"
            β_model, weighted_model, naive_model, no_synth_model = init_stan_models(n_samples, n_warmup)
        end

        for iter in 1:iterations

            println("Loading data...")
            labels, real_data, synth_data = load_data(name, label, ε)
            println("Distributing work...")

            progress_pmap(1:total_steps, progress=p) do i

                fold = ((i - 1) % folds)
                real_α, synth_α = αs[Int(ceil(i / folds))]
                X_real, y_real, X_synth, y_synth, X_valid, y_valid = fold_α(
                    real_data, synth_data, real_α, synth_α,
                    fold, folds, labels
                )
                metric, initial_θ = init_run(
                    θ_dim, λ, X_real, y_real, X_synth, y_synth, β
                )
                βw_calib = weight_calib(X_synth, y_synth, β, initial_θ)
                if isnan(βw_calib)
                    βw_calib = βw
                end
                println(βw_calib)

                if sampler == "Stan"

                    data = Dict(
                        "f" => θ_dim - 1,
                        "a" => size(X_real)[1],
                        "X_real" => X_real[:, 2:end],
                        "y_real" => y_real,
                        "b" => size(X_synth)[1],
                        "X_synth" => X_synth[:, 2:end],
                        "y_synth" => y_synth,
                        "c" => size(X_valid)[1],
                        "X_test" => X_valid[:, 2:end],
                        "y_test" => y_valid,
                        "w" => w,
                        "beta" => β,
                        "beta_w" => βw_calib
                    )
                    rc = stan_sample(
                        β_model;
                        data=data,
                        n_chains=n_chains,
                        init=[Dict("alpha" => initial_θ[1], "coefs" => initial_θ[2:end])],
                        cores=1
                    )
                    if success(rc)
                        θ_β = mean(read_samples(β_model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                        auc_β, ll_β, bf_β = evaluate_stan(X_valid, y_valid, θ_β)
                    end
                    rc = stan_sample(
                        weighted_model;
                        data=data,
                        n_chains=n_chains,
                        init=[Dict("alpha" => initial_θ[1], "coefs" => initial_θ[2:end])],
                        cores=1
                    )
                    if success(rc)
                        θ_weighted = mean(read_samples(weighted_model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                        auc_weighted, ll_weighted, bf_weighted = evaluate_stan(X_valid, y_valid, θ_weighted)
                    end
                    rc = stan_sample(
                        naive_model;
                        data=data,
                        n_chains=n_chains,
                        init=[Dict("alpha" => initial_θ[1], "coefs" => initial_θ[2:end])],
                        cores=1
                    )
                    if success(rc)
                        θ_naive = mean(read_samples(naive_model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                        auc_naive, ll_naive, bf_naive = evaluate_stan(X_valid, y_valid, θ_naive)
                    end
                    rc = stan_sample(
                        no_synth_model;
                        data=data,
                        n_chains=n_chains,
                        init=[Dict("alpha" => initial_θ[1], "coefs" => initial_θ[2:end])],
                        cores=1
                    )
                    if success(rc)
                        θ_no_synth = mean(read_samples(no_synth_model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                        auc_no_synth, ll_no_synth, bf_no_synth = evaluate_stan(X_valid, y_valid, θ_no_synth)
                    end

                elseif sampler == "AHMC"

                    # Define log posteriors and gradients of them
                    ℓπ_β, ∂ℓπ∂θ_β = (
                        ℓπ_beta(σ, β, βw_calib, X_real, y_real, X_synth, y_synth),
                        ∂ℓπ∂θ_beta(σ, β, βw_calib, X_real, y_real, X_synth, y_synth)
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
                    println("RUNNING BETA")
                    hamiltonian_β, proposal_β, adaptor_β = setup_run(
                        ℓπ_β,      # 61.805 μs (10.47% GC)  memory estimate:  32.80 KiB  allocs estimate:  7
                        ∂ℓπ∂θ_β,   # 1.274 ms (2.00% GC)  memory estimate:  564.16 KiB  allocs estimate:  38
                        metric,
                        initial_θ,
                        use_ad=use_ad
                    )
                    samples_β, stats_β = sample(
                        hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
                        drop_warmup=true, progress=show_progress, verbose=show_progress
                    )
                    auc_β, ll_β, bf_β = evalu(X_valid, y_valid, samples_β)

                    # KLD WEIGHTED
                    hamiltonian_weighted, proposal_weighted, adaptor_weighted = setup_run(
                        ℓπ_weighted,     # 36.188 μs (8.75% GC)  memory estimate:  29.80 KiB  allocs estimate:  6
                        ∂ℓπ∂θ_weighted,  # 82.259 μs (9.74% GC)  memory estimate:  64.38 KiB  allocs estimate:  20
                        metric,
                        initial_θ,
                        use_ad=use_ad
                    )
                    samples_weighted, stats_weighted = sample(
                        hamiltonian_weighted, proposal_weighted, initial_θ, n_samples, adaptor_weighted, n_warmup;
                        drop_warmup=true, progress=show_progress, verbose=show_progress
                    )
                    auc_weighted, ll_weighted, bf_weighted = evalu(X_valid, y_valid, samples_weighted)

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
                    auc_naive, ll_naive, bf_naive = evalu(X_valid, y_valid, samples_naive)

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
                    auc_no_synth, ll_no_synth, bf_no_synth = evalu(X_valid, y_valid, samples_no_synth)

                end
                # bf_matrix = create_bayes_factor_matrix([bf_β, bf_weighted, bf_naive, bf_no_synth])
                # results[i, :] =
                # bayes_factors[:, :, i] = bf_matrix
                open("$(path)/$(name)_$(t)_out.csv", "a") do io
                    write(io, "$(iter),$(fold),$(real_α),$(synth_α),$(auc_β),$(auc_weighted),$(auc_naive),$(auc_no_synth),$(ll_β),$(ll_weighted),$(ll_naive),$(ll_no_synth)\n")
                end
            end
        end
    else
        println("Loading data...")
        labels, real_data, synth_data = load_data(name, label, ε)
        println("Beginning experiment...")
        @showprogress for i in 1:total_steps

            fold = ((i - 1) % folds)
            real_α, synth_α = αs[Int(ceil(i / folds))]
            X_real, y_real, X_synth, y_synth, X_valid, y_valid = fold_α(
                real_data, synth_data, real_α, synth_α,
                fold, folds, labels
            )
            metric, initial_θ = init_run(
                θ_dim, λ, X_real, y_real, X_synth, y_synth, β
            )

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
                use_ad=true
            )
            samples_β, stats_β = sample(
                hamiltonian_β, proposal_β, initial_θ, n_samples, adaptor_β, n_warmup;
                drop_warmup=true, progress=show_progress, verbose=show_progress
            )
            auc_β, ll_β, bf_β = evalu(X_valid, y_valid, samples_β)

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
            auc_weighted, ll_weighted, bf_weighted = evalu(X_valid, y_valid, samples_weighted)

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
            auc_naive, ll_naive, bf_naive = evalu(X_valid, y_valid, samples_naive)

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
            auc_no_synth, ll_no_synth, bf_no_synth = evalu(X_valid, y_valid, samples_no_synth)

            # bf_matrix = create_bayes_factor_matrix([bf_β, bf_weighted, bf_naive, bf_no_synth])
            # results[i, :] = [real_α, synth_α, auc_β, auc_weighted, auc_naive, auc_no_synth, ll_β, ll_weighted, ll_naive, ll_no_synth]
            # bayes_factors[:, :, i] = bf_matrix
            # CSV.write("src/logistic_regression/outputs/results___$(t).csv", create_results_df(results))
        end
    end

    # Record the results in csv and JLD objects
    # save("src/logistic_regression/outputs/all_out___$(t).jld", "data", outs)
    # results_df = create_results_df(results)
    # CSV.write("src/creditcard/outputs/results___$(t).csv", results_df)
    # save("src/creditcard/outputs/bayes_factors___$(t).jld", "data", bayes_factors)
    println("Done")
end

main()

for i in workers()
    rmprocs(i)
end
