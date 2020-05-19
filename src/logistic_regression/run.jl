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
using Random
using StatsFuns: log1pexp, log2π
using MLJBase: auc
using StanSample
include("../common/utils.jl")
include("../common/weight_calibration.jl")
include("distributions.jl")
include("loss.jl")
include("evaluation.jl")
include("init.jl")

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
    using StatsFuns: log1pexp, log2π
    using MLJBase: auc
    using StanSample
    include("src/common/utils.jl")
    include("src/common/weight_calibration.jl")
    include("src/logistic_regression/distributions.jl")
    include("src/logistic_regression/loss.jl")
    include("src/logistic_regression/evaluation.jl")
    include("src/logistic_regression/init.jl")
end


function main()

    args = parse_cl()
    path, dataset, label, ε = args["path"], args["dataset"], args["label"], args["epsilon"]
    iterations, folds, split = args["iterations"], args["folds"], args["split"]
    use_ad, distributed, sampler, no_shuffle = args["use_ad"], args["distributed"], args["sampler"], args["no_shuffle"]
    # path, dataset, label, ε, iterations, folds, split, distributed, use_ad, sampler, no_shuffle = "src", "uci_heart", "target", "6.0", 1, 5, 1.0, false, false, "Stan", false
    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")

    println("Setting up experiment...")
    w = 0.5
    β = 0.5
    βw = 1.15
    σ = 50.0
    λ = 1.0
    real_αs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
    synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
    αs = get_conditional_pairs(real_αs, synth_αs)
    num_αs = size(αs)[1]
    iter_steps = num_αs * folds
    total_steps = iter_steps * iterations
    n_samples, n_warmup = 10000, 2000
    model_names = [
        "beta", "weighted", "naive", "no_synth"
    ]
    if sampler == "Stan"
        mkpath("$(@__DIR__)/tmp/")
        models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(model_names, n_samples, n_warmup; dist = distributed))
            end
        )
    end
    n_chains = 1
    show_progress = true

    io = open("$(path)/$(dataset)_$(t)_out.csv", "w")
    write(io, "iter,fold,real_α,synth_α,beta_auc,beta_ll,beta_bf,weighted_auc,weighted_ll,weighted_bf,naive_auc,naive_ll,naive_bf,no_synth_auc,no_synth_ll,no_synth_bf\n")
    close(io)

    println("Loading data...")
    labels, unshuffled_real_data, unshuffled_synth_data = load_data(dataset, label, ε)
    θ_dim = size(unshuffled_real_data)[2] - 1
    unshuffled_real_data[:, labels[1]] = (2 .* unshuffled_real_data[:, labels[1]]) .- 1
    unshuffled_synth_data[:, labels[1]] = (2 .* unshuffled_synth_data[:, labels[1]]) .- 1
    c = classes(categorical(unshuffled_real_data[:, labels[1]])[1])
    println("Distributing work...")
    p = Progress(total_steps)

    progress_pmap(1:total_steps, progress=p) do i

        iter = Int(ceil(i / iter_steps))
        iter_i = i - (iter - 1) * iter_steps
        fold = ((iter_i - 1) % folds)
        Random.seed!(iter)
        real_data = unshuffled_real_data[shuffle(axes(unshuffled_real_data, 1)), :]
        synth_data = unshuffled_synth_data[shuffle(axes(unshuffled_synth_data, 1)), :]
        real_α, synth_α = αs[Int(ceil(iter_i / folds))]

        println("Worker $(myid()) on iter $(iter), step $(iter_i) with real alpha = $(real_α) and synthetic alpha = $(synth_α)...")
        X_real, y_real, X_synth, y_synth, X_valid, y_valid = fold_α(
            real_data, synth_data, real_α, synth_α,
            fold, folds, labels
        )
        initial_θ = init_run(
            λ, X_real, y_real, X_synth, y_synth, β
        )
        # βw_calib = weight_calib(X_synth, y_synth, β, initial_θ, βloss, ∇βloss, Hβloss)
        # if isnan(βw_calib)
        #     βw_calib = βw
        # end
        βw_calib = βw
        # println(βw_calib)
        evaluations = []

        if sampler == "Stan"

            data = Dict(
                "f" => θ_dim - 1,
                "a" => size(X_real)[1],
                "X_real" => X_real[:, 2:end],
                "y_real" => Int.((y_real .+ 1) ./ 2),
                "b" => size(X_synth)[1],
                "X_synth" => X_synth[:, 2:end],
                "y_synth" => Int.((y_synth .+ 1) ./ 2),
                "w" => w,
                "beta" => β,
                "beta_w" => βw_calib
            )
            for (name, model) in models[myid()]

                println("Running $(name)...")
                rc = stan_sample(
                    model;
                    data=data,
                    n_chains=n_chains,
                    init=[Dict("alpha" => initial_θ[1], "coefs" => initial_θ[2:end])],
                    cores=1
                )
                if success(rc)
                    samples = mean(read_samples(model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                    auc_score, ll, bf = evaluate_samples(X_valid, y_valid, samples, c)
                    append!(evaluations, [auc_score, ll, bf])
                else
                    append!(evaluations, [NaN, NaN, NaN])
                end

            end

        elseif sampler == "AHMC"

            # Define log posteriors and gradients of them
            models = init_ahmc_models(
                X_real, y_real, X_synth, y_synth, σ, w, βw_calib, β, initial_θ
            )

            for (name, model) in models

                println("Running $(name)...")
                metric = DiagEuclideanMetric(θ_dim)
                hamiltonian, proposal, adaptor = setup_run(
                    model.ℓπ,
                    model.∇ℓπ,
                    metric,
                    initial_θ,
                    use_ad=use_ad
                )
                samples, stats = sample(
                    hamiltonian, proposal, initial_θ, n_samples, adaptor, n_warmup;
                    drop_warmup=true, progress=show_progress, verbose=show_progress
                )
                auc_score, ll, bf = evaluate_samples(X_valid, y_valid, hcat(samples...)', c)
                append!(evaluations, [auc_score, ll, bf])

            end

        elseif sampler == "Turing"

            models = init_turing_models(
                X_real, y_real, X_synth, y_synth, σ, w, βw_calib, β
            )

            for (name, model) in models

                println("Running $(name)...")
                varinfo = Turing.VarInfo(model)
                model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((θ = initial_θ,)))
                init_θ = varinfo[Turing.SampleFromPrior()]
                chn = sample(model, Turing.NUTS(), n_samples, init_theta = init_θ)
                auc_score, ll, bf = evaluate_samples(X_valid, y_valid, Array(chn), c)
                append!(evaluations, [auc_score, ll, bf])

            end

        end

        open("$(path)/$(dataset)_$(t)_out.csv", "a") do io
            write(io, "$(iter),$(fold),$(real_α),$(synth_α),$(join(evaluations, ','))\n")
        end
    end

    println("Done")
end

main()

for i in workers()
    rmprocs(i)
end
