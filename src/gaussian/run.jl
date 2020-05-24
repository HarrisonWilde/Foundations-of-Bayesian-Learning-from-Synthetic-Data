using ClusterManagers
using Distributed
# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
println("We are all connected and ready.")
for i in workers()
    host, pid = fetch(@spawnat i (gethostname(), getpid()))
    println(host, pid)
end
using Bijectors
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
using QuadGK
using Roots
include("../common/utils.jl")
include("../common/init.jl")
include("distributions.jl")
include("loss.jl")
include("evaluation.jl")
include("init.jl")
include("weight_calibration.jl")

@everywhere begin
    using Distributed
    using Bijectors
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
    using StatsFuns: invsqrt2π, log2π, sqrt2
    using MLJBase: auc
    using StanSample
    using CmdStan
    using QuadGK
    using Roots
    include("src/common/utils.jl")
    include("src/common/init.jl")
    include("src/gaussian/init.jl")
    include("src/gaussian/distributions.jl")
    include("src/gaussian/evaluation.jl")
    include("src/gaussian/loss.jl")
    include("src/gaussian/weight_calibration.jl")
end


function main()

    args = parse_cl()
    λs = args["scales"]
    path, iterations = args["path"], args["iterations"]
    use_ad, distributed, sampler, no_shuffle = args["use_ad"], args["distributed"], args["sampler"], args["no_shuffle"]
    # λs, path, iterations, distributed, use_ad, sampler, no_shuffle = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], ".", 5, false, false, "CmdStan", false
    experiment_type = "gaussian"
    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    out_path = "$(path)/src/$(experiment_type)/outputs/$(t)"
    mkpath(out_path)

    println("Setting up experiment...")
    w = 0.5
    β = 0.5
    βw = 1.15
    μ = 0.
    σ = 1.
    αₚ, βₚ, μₚ, σₚ = 3., 5., 1., 1.
    real_ns = vcat([1, 5], collect(1:5) .^ 2 .* 10)
    synth_ns = vcat(collect(0:2:9), collect(10:5:29), collect(30:10:149), collect(150:25:250))
    unseen_n = 500
    ns = get_conditional_pairs(real_ns, synth_ns, max_sum=maximum(real_ns))
    num_ns = length(ns)
    num_λs = length(λs)
    iter_steps = num_ns * num_λs
    total_steps = iter_steps * iterations
    n_samples, n_warmup = 12500, 2500
    nchains = 3
    target_acceptance_rate = 0.8
    metrics = ["ll", "kld", "wass"]
    model_names = [
        "beta", "weighted", "naive", "no_synth", "beta_all", "noise_aware"
    ]
    name_metrics = join(["$(name)_$(metric)" for name in model_names for metric in metrics], ",")
    if (sampler == "CmdStan") | (sampler == "Stan")
        mkpath("$(path)/src/$(experiment_type)/tmp$(sampler)/")
        models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(path, sampler, model_names, experiment_type, target_acceptance_rate, nchains, n_samples, n_warmup; dist = distributed))
            end
        )
    end
    show_progress = true

    @everywhere begin
        open("$($out_path)/$(myid())_out.csv", "w") do io
            write(io, "iter,scale,real_α,synth_α,$($name_metrics)\n")
        end
    end

    println("Generating data...")
    dgp = Distributions.Normal(μ, σ)
    all_real_data = rand(dgp, (iterations, maximum(real_ns)))
    pre_contam_data = rand(dgp, (iterations, maximum(real_ns)))
    all_synth_data = vcat([pre_contam_data + rand(Laplace(0, λ), (iterations, maximum(real_ns))) for λ in λs]...)
    all_unseen_data = rand(dgp, (iterations, unseen_n))
    println("Distributing work...")
    p = Progress(total_steps)

    # βw_calib = weight_calib(
    #     all_synth_data,
    #     β
    # )
    # if isnan(βw_calib)
    #     βw_calib = βw
    # end
    βw_calib = βw
    # println(βw_calib)

    progress_pmap(1:total_steps, progress=p) do i

        iter = Int(ceil(i / iter_steps))
        iter_i = i - (iter - 1) * iter_steps
        noise_scale = Int(ceil(iter_i / num_ns))
        noise_scale_i = iter_i - (noise_scale - 1) * num_ns
        real_n, synth_n = ns[noise_scale_i]
        real_data = all_real_data[iter, 1:real_n]
        synth_data = all_synth_data[noise_scale * iter, 1:synth_n]
        unseen_data = all_unseen_data[iter, :]
        λ = λs[noise_scale]
        println("Worker $(myid()) on iter $(iter), step $(iter_i) with scale = $(λ), real n = $(real_n) and synthetic n = $(synth_n)...")

        evaluations = []

        if sampler == "CmdStan"

            data = Dict(
                "n" => real_n,
                "y1" => real_data,
                "m" => synth_n,
                "y2" => synth_data,
                "p_mu" => μₚ,
                "p_alpha" => αₚ,
                "p_beta" => βₚ,
                "hp" => σₚ,
                "scale" => λ,
                "beta" => β,
                "beta_w" => βw_calib,
                "w" => w,
                "lambda" => λ
            )
            @time for (name, model) in models[myid()]

                println("Running $(name)...")
                rc, chn, _ = stan(
                    model,
                    data
                )
                if rc == 0
                    samples = Array(chn)[:, 1:2]
                    @show size(samples)
                    @show mean(samples, dims=1)
                    ll, kld, wass = evaluate_samples(unseen_data, dgp, samples)
                    append!(evaluations, [ll, kld, wass])
                else
                    append!(evaluations, [NaN, NaN, NaN])
                end

            end

        elseif sampler == "Stan"

            data = Dict(
                :n => real_n,
                :y1 => real_data,
                :m => synth_n,
                :y2 => synth_data,
                :p_mu => μₚ,
                :p_alpha => αₚ,
                :p_beta => βₚ,
                :hp => σₚ,
                :scale => λ,
                :beta => β,
                :beta_w => βw_calib,
                :w => w,
                :lambda => λ
            )
            for (name, model) in models[myid()]

                println("Running $(name)...")
                rc = stan_sample(
                    model;
                    data=data
                )
                if success(rc)
                    samples = mean(read_samples(model)[:, 1:2, :], dims=3)[:, :, 1]
                    ll, kld, wass = evaluate_samples(unseen_data, dgp, samples)
                    append!(evaluations, [ll, kld, wass])
                else
                    append!(evaluations, [NaN, NaN, NaN])
                end

            end

        elseif sampler == "AHMC"

            # Define log posteriors and gradients of them
            b, models = init_ahmc_models(
                real_data, synth_data, w, βw_calib, β, λ, αₚ, βₚ, μₚ, σₚ
            )

            @time for (name, model) in models

                println("Running $(name)...")
                chains = map(1:nchains) do i
                    metric = DiagEuclideanMetric(2)
                    initial_θ = rand(2)
                    hamiltonian, proposal, adaptor = setup_run(
                        model,
                        metric,
                        initial_θ;
                        target_acceptance_rate = target_acceptance_rate
                    )
                    chn, _ = sample(
                        hamiltonian, proposal, initial_θ, n_samples, adaptor, n_warmup;
                        drop_warmup=true, progress=show_progress, verbose=show_progress
                    )
                    chn
                end
                chains = hcat(b.(vcat(chains...))...)'
                @show size(chains)
                @show mean(chains, dims=1)
                ll, kl, wass = evaluate_samples(unseen_data, dgp, chains)
                append!(evaluations, [ll, kl, wass])

            end

        elseif sampler == "Turing"

            models = init_turing_models(
                real_data, synth_data, w, βw_calib, β, λ, αₚ, βₚ, μₚ, σₚ
            )

            @time for (name, model) in models

                println("Running $(name)...")
                chains = map(1:nchains) do i
                    chn = sample(model, Turing.NUTS(n_warmup, target_acceptance_rate), n_samples)
                    Array(chn)
                end
                @show size(vcat(chains...))
                @show mean(vcat(chains...), dims=1)
                ll, kl, wass = evaluate_samples(unseen_data, dgp, vcat(chains...))
                append!(evaluations, [ll, kl, wass])

            end

        end

        open("$(out_path)/$(myid())_out.csv", "a") do io
            write(io, "$(iter),$(λ),$(real_n),$(synth_n),$(join(evaluations, ','))\n")
        end
    end

    println("Done")
end

main()

for i in workers()
    rmprocs(i)
end
