using Distributed
# using ClusterManagers
# # # addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])))
# println("We are all connected and ready.")
# for i in workers()
#     host, pid = fetch(@spawnat i (gethostname(), getpid()))
#     println(host, pid)
# end
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
using Random
using StatsFuns: invsqrt2π, log2π, sqrt2
using MLJBase: auc
using StanSample
using CmdStan
using DataStructures
# include("../common/utils.jl")
# include("../common/init.jl")
# include("distributions.jl")
# include("init.jl")

@everywhere begin
    using Distributed
    using ForwardDiff
    using LinearAlgebra
    using CSV
    using DataFrames
    using AdvancedHMC
    using Distributions
    using Turing
    using Zygote
    using Random
    using MCMCChains
    using JLD
    using MLJ
    using Optim
    using MLJLinearModels
    using Dates
    using ProgressMeter
    using SharedArrays
    using SpecialFunctions
    using StatsFuns: invsqrt2π, log2π, sqrt2
    using MLJBase: auc
    using StanSample
    using CmdStan
    using DataStructures
    # include("src/common/utils.jl")
    # include("src/common/init.jl")
    # include("src/gaussian/init.jl")
    # include("src/gaussian/distributions.jl")
end

path, dataset, label, ε = ".", "gcse", "course", "6.0"
iterations, folds, split = 1, 5, 1.0
distributed, use_ad, sampler, no_shuffle = false, false, "Turing", false
experiment_type = "regression"
w = 0.5
β = 0.5
βw = 1.25
real_αs = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
n_samples, n_warmup = 5000, 1000
nchains = 1
target_acceptance_rate = 0.8
metrics = ["rmse", "ll"]
model_names = [
    "beta", "weighted", "naive", "no_synth"
]

function run_experiment()

    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    out_path = "$(path)/src/$(experiment_type)/outputs/$(dataset)_$(t)"
    mkpath(out_path)

    println("Setting up experiment...")
    predictors = [:female]
    groups = [:school]

    αs = get_conditional_pairs(real_αs, synth_αs)
    num_αs = size(αs)[1]
    iter_steps = num_αs * folds
    total_steps = iter_steps * iterations

    name_metrics = join(["$(name)_$(metric)" for name in model_names for metric in metrics], ",")
    if (sampler == "CmdStan") | (sampler == "Stan")
        mkpath("$(path)/src/$(experiment_type)/tmp$(sampler)/")
        models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(path, sampler, model_names, experiment_type, target_acceptance_rate, nchains, n_samples, n_warmup; dist = distributed))
            end
        )
    end
    show_progress = distributed

    @everywhere begin
        open("$($out_path)/$(myid())_out.csv", "w") do io
            write(io, "iter,fold,real_α,synth_α,$($name_metrics)\n")
        end
    end

    println("Loading data...")
    labels, unshuffled_real_data, unshuffled_synth_data = load_data(dataset, label, ε)

    # Standardise data
    unshuffled_real_data[!, :course] = (unshuffled_real_data[!, :course] .- mean(unshuffled_real_data[!, :course])) / std(unshuffled_real_data[!, :course])
    unshuffled_synth_data[!, :course] = (unshuffled_synth_data[!, :course] .- mean(unshuffled_synth_data[!, :course])) / std(unshuffled_synth_data[!, :course])

    # Declare priors
    θ_dim = length(predictors) + 1
    μₚ, σₚ, αₚ, βₚ, νₚ, Σₚ = 0., 1., 0.1, 0.1, 3., Matrix(Diagonal([2. for i in 1:θ_dim]))
    println("Distributing work...")
    p = Progress(total_steps)

    # try
    #     βw_calib = weight_calib(
    #         Matrix(unshuffled_synth_data[:, Not(labels)]),
    #         Int.(unshuffled_synth_data[:, labels[1]]),
    #         β, λ
    #     )
    #     if isnan(βw_calib)
    #         βw_calib = βw
    #     elseif βw_calib > 5
    #         βw_calib = 5
    #     elseif βw_calib < 0.5
    #         βw_calib = 0.5
    #     end
    # catch
    #     βw_calib = βw
    # end
    # @show βw_calib
    βw_calib = βw

    progress_pmap(100:total_steps, progress=p) do i

        iter = Int(ceil(i / iter_steps))
        iter_i = i - (iter - 1) * iter_steps
        fold = ((iter_i - 1) % folds)
        Random.seed!(iter)
        real_data = unshuffled_real_data[shuffle(axes(unshuffled_real_data, 1)), :]
        synth_data = unshuffled_synth_data[shuffle(axes(unshuffled_synth_data, 1)), :]
        real_α, synth_α = αs[Int(ceil(iter_i / folds))]
        println("Worker $(myid()) on iter $(iter), step $(iter_i) with real alpha = $(real_α) and synthetic alpha = $(synth_α)...")

        X_real, y_real, groups_real, X_synth, y_synth, groups_synth, X_valid, y_valid, groups_valid = fold_α(
            real_data, synth_data, real_α, synth_α,
            fold, folds, labels;
            predictors = predictors, groups = groups,
            add_intercept = true, continuous_y = true
        )

        unique_groups = unique(vcat(groups_real, groups_synth, groups_valid))
        for i in 1:length(unique_groups)

            groups_real[findall(x -> x == unique_groups[i], groups_real)] .= i
            groups_synth[findall(x -> x == unique_groups[i], groups_synth)] .= i
            groups_valid[findall(x -> x == unique_groups[i], groups_valid)] .= i

        end
        nₛ = length(unique_groups)
        nₚ = θ_dim

        # @show X_real, y_real, groups_real, X_synth, y_synth, groups_synth, X_valid, y_valid, groups_valid, unique_groups

        # initial_θ = init_run(
        #     λ, X_real, y_real, schools_real, β
        # )
        evaluations = []

        if sampler == "CmdStan"

            data = Dict(
                "n_real" => size(X_real)[1] == 0 ? 1 : size(X_real)[1],
                "n_synth" => size(X_synth)[1] == 0 ? 1 : size(X_synth)[1],
                "n_groups" => size(X_synth)[1] == 0 & size(X_real)[1] == 0 ? 1 : nₛ,
                "n_theta" => θ_dim,
                "y_real" => size(X_real)[1] == 0 ? [50.] : y_real,
                "y_synth" => size(X_synth)[1] == 0 ? [50.] : y_synth,
                "X_real" => size(X_real)[1] == 0 ? zeros(1, size(X_real)[2]) : X_real,
                "X_synth" => size(X_synth)[1] == 0 ? zeros(1, size(X_synth)[2]) : X_synth,
                "schools_real" => size(X_real)[1] == 0 ? [1] : groups_real,
                "schools_synth" => size(X_real)[1] == 0 ? [1] : groups_synth,
                "p_mu" => μₚ,
                "p_sigma" => σₚ,
                "p_alpha" => αₚ,
                "p_beta" => βₚ,
                "p_nu" => νₚ,
                "p_Sigma" => Σₚ,
                "w" => w,
                "beta" => β,
                "beta_w" => βw_calib,
                "flag_real" => size(X_synth)[1] == 0 ? 1 : 0,
                "flag_synth" => size(X_synth)[1] == 0 ? 1 : 0
            )
            # init = Dict(
            #     "alpha" => initial_θ[1],
            #     "coefs" => initial_θ[2:end]
            # )
            @time for (name, model) in models[myid()]

                println("Running $(name)...")
                try
                    _, chn, _ = stan(
                        model,
                        data;
                        # init=init
                    )
                    samples = Array(chn)[:, 1:θ_dim]
                    @show mean(samples, dims=1)
                    mse, ll, bf = evaluate_samples(X_valid, y_valid, samples)
                    append!(evaluations, [mse, ll, bf])
                catch
                    append!(evaluations, [NaN, NaN, NaN])
                end

            end

        elseif sampler == "AHMC"

            # Define log posteriors and gradients of them
            models = init_ahmc_models(
                X_real, y_real, X_synth, y_synth, σ, w, βw_calib, β, initial_θ
            )

            @time for (name, model) in models

                println("Running $(name)...")
                metric = DiagEuclideanMetric(θ_dim)
                hamiltonian, proposal, adaptor = setup_run(
                    model.ℓπ,
                    metric,
                    initial_θ;
                    ∂ℓπ∂θ = model.∇ℓπ,
                    target_acceptance_rate = target_acceptance_rate
                )
                chn, _ = sample(
                    hamiltonian, proposal, initial_θ, n_samples, adaptor, n_warmup;
                    drop_warmup=true, progress=show_progress, verbose=show_progress
                )
                @show mean(hcat(chn...)', dims=1)
                auc_score, ll, bf = evaluate_samples(X_valid, y_valid, hcat(chn...)', c)
                append!(evaluations, [auc_score, ll, bf])

            end

        elseif sampler == "Turing"

            models = init_turing_models(
                y_real, X_real, groups_real,
            	y_synth, X_synth, groups_synth,
            	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
            	nₚ, nₛ, β, βw, w
            )

            @time for (name, model) in models

                println("Running $(name)...")
                # varinfo = Turing.VarInfo(model)
                # model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((θ = initial_θ,)))
                # init_θ = varinfo[Turing.SampleFromPrior()]
                chn = sample(
                    model,
                    Turing.NUTS(n_warmup, target_acceptance_rate),
                    n_samples,
                    # init_theta = init_θ
                )
                @show mean(Array(chn), dims=1)
                auc_score, ll, bf = evaluate_samples(X_valid, y_valid, Array(chn), c)
                append!(evaluations, [auc_score, ll, bf])

            end

        end

        open("$(out_path)/$(myid())_out.csv", "a") do io
            write(io, "$(iter),$(fold),$(real_α),$(synth_α),$(join(evaluations, ','))\n")
        end
    end

    println("Done")
end

run_experiment()
