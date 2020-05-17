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
    λs = args["scales"]
    path, iterations = args["path"], args["iterations"]
    use_ad, distributed, sampler, no_shuffle = args["use_ad"], args["distributed"], args["sampler"], args["no_shuffle"]
    # λs, path, iterations, distributed, use_ad, sampler, no_shuffle = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], "src", 5, false, false, "Stan", false
    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")

    println("Setting up experiment...")
    w = 0.5
    β = 0.5
    βw = 1.15
    μ = 0.
    σ = 1.
    real_ns = vcat([1, 5], collect(1:5) .^ 2 .* 10)
    synth_ns = vcat(collect(0:2:9), collect(10:5:29), collect(30:10:149), collect(150:25:250))
    unseen_n = 500
    ns = get_conditional_pairs(real_ns, synth_ns, max_sum=maximum(real_ns))
    num_ns = length(ns)
    num_λs = length(λs)
    iter_steps = num_ns * num_λs
    total_steps = iter_steps * iterations
    metrics = ["ll"]
    if distributed
        @everywhere n_samples, n_warmup = 5000, 1000
        @everywhere model_names = [
            "beta", "weighted", "naive", "no_synth", "beta_all", "noise_aware"
        ]
    else
        n_samples, n_warmup = 5000, 1000
        model_names = [
            "beta", "weighted", "naive", "no_synth", "beta_all", "noise_aware"
        ]
    end
    if (sampler == "Stan") & distributed
        @everywhere models = init_stan_models(model_names, n_samples, n_warmup; dist = true)
    elseif sampler == "Stan"
        models = init_stan_models(model_names, n_samples, n_warmup; dist = false)
    end
    n_chains = 3
    show_progress = true

    io = open("$(path)/$(dataset)_$(t)_out.csv", "w")
    name_metrics = join(["$(name)_$(metric)" for name in model_names for metric in metrics], ",")
    write(io, "iter,scale,real_α,synth_α,$(name_metrics)\n")
    close(io)

    println("Generating data...")
    dgp = Normal(μ, σ)
    all_real_data = rand(dgp, (iterations, maximum(real_ns)))
    pre_contam_data = rand(dgp, (iterations, maximum(real_ns)))
    all_synth_data = vcat([pre_contam_data + rand(Laplace(0, λ), (iterations, maximum(real_ns))) for λ in λs]...)
    unseen_data = rand(dgp, (iterations, unseen_n))
    println("Distributing work...")
    p = Progress(total_steps)

    progress_pmap(1:total_steps, progress=p) do i

        iter = Int(ceil(i / iter_steps))
        iter_i = i - (iter - 1) * iter_steps
        scale = Int(ceil(iter_i / num_ns))
        scale_i = iter_i - (scale - 1) * num_ns
        real_n, synth_n = ns[scale_i]
        real_data = all_real_data[iter, 1:real_n]
        synth_data = all_synth_data[scale * iter, 1:synth_n]
        λ = λs[scale]
        println("Worker $(myid()) on iter $(iter), step $(iter_i) with scale = $(λ), real n = $(real_n) and synthetic n = $(synth_n)...")
        # βw_calib = weight_calib(synth_data, β, βloss, ∇βloss, Hβloss)
        # if isnan(βw_calib)
        #     βw_calib = βw
        # end
        βw_calib = βw
        println(βw_calib)
        evaluations = []

        if sampler == "Stan"

            data = Dict(
                "n" => real_n,
                "y1" => real_data,
                "m" => synth_n,
                "y2" => synth_data,
                "p_mu" => 1,
                "p_alpha" => 3,
                "p_beta" => 5,
                "hp" => 1,
                "scale" => λ,
                "beta" => β,
                "beta_w" => βw_calib,
                "w" => w,
            )
            for (name, model) in models

                println("Running $(name)...")
                rc = stan_sample(
                    model;
                    data=data,
                    n_chains=n_chains,
                    cores=1
                )
                if success(rc)
                    samples = mean(read_samples(model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                    ll, kld, wass = evaluate_samples(unseen_data, samples, c)
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
                chn = sample(model, Turing.NUTS(), n_samples, init_theta = init_\theta)
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
