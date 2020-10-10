using ClusterManagers
using Distributed
# addprocs(6)
# @everywhere begin
#     using Pkg; Pkg.activate("."); Pkg.instantiate()
# end

# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
# addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]), enable_threaded_blas=false, topology=:master_worker)
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
println("We are all connected and ready.")
for i in workers()
    host, pid = fetch(@spawnat i (gethostname(), getpid()))
    println(host, pid)
end

using AdvancedHMC
using ArgParse
using Bijectors
using CmdStan
using CSV
using DataFrames
using DataStructures
using Dates
using Distributions
using ForwardDiff
using HypothesisTests
using IterTools
using JLD
using LinearAlgebra
using MCMCChains
using MLJ
using MLJBase: auc
using MLJLinearModels
using Optim
using Plots
using ProgressMeter
using QuadGK
using Random
using Roots
using SpecialFunctions
using StanSample
using Statistics
using StatsFuns: invsqrt2π, log2π, sqrt2, log1pexp
using Turing
using Zygote
include("src/distributions.jl")
include("src/evaluation.jl")
include("src/init.jl")
include("src/loss.jl")
include("src/utils.jl")
include("src/weight_calibration.jl")

@everywhere begin
    using Distributed
    using AdvancedHMC
    using ArgParse
    using Bijectors
    using CmdStan
    using CSV
    using DataFrames
    using DataStructures
    using Dates
    using Distributions
    using ForwardDiff
    using HypothesisTests
    using JLD
    using LinearAlgebra
    using MCMCChains
    using MLJ
    using MLJBase: auc
    using MLJLinearModels
    using Optim
    using Plots
    using ProgressMeter
    using QuadGK
    using Random
    using Roots
    using SpecialFunctions
    using StanSample
    using Statistics
    using StatsFuns: invsqrt2π, log2π, sqrt2, log1pexp
    using Turing
    using Zygote
    include("src/distributions.jl")
    include("src/evaluation.jl")
    include("src/init.jl")
    include("src/loss.jl")
    include("src/utils.jl")
    include("src/weight_calibration.jl")
end

# @everywhere Turing.turnprogress(false)
# args = Dict(
#     "experiment_type" => "gaussian",
#     "show_progress" => true,
#     "iterations" => 100,
#     "n_samples" => 10000,
#     "n_warmup" => 0,
#     "n_chains" => 1,
#     "sampler" => "Turing",
#     "scales" => [0.006, 0.06, 0.6, 1.0, 6.0, 60.0, 600.0],
#     "betas" => [0.75],
#     "beta_weights" => [1.25],
#     "calibrate_beta_weight" => false,
#     "weights" => [1.0],
#     "metrics" => ["ll", "kld", "wass"],
#     "model_names" => ["weighted"],
#     "real_ns" => [0],
#     "real_n_range" => [],
#     "synth_ns" => [],
#     "synth_n_range" => [0, 1, 100],
#     "n_unseen" => 100,
#     "algorithm" => "basic",
#     "mu_p" => 3.0,
#     "sigma_p" => 30.0,
#     "alpha_p" => 2.0,
#     "beta_p" => 4.0,
#     "nu_p" => -1,
#     "Sigma_p" => -1,
#     "path" => ".",
#     "folds" => 5,
#     "split" => 1.0,
#     "use_ad" => false,
#     "distributed" => false,
#     "target_acceptance" => 0.8,
#     "mu" => 0.0,
#     "sigma" => 1.0,
#     "id" => "noise_demo",
#     "seed" => 1,
#     "alphas" => [],
#     "fn" => 0
# )

# args = Dict(
#     "experiment_type" => "logistic_regression",
#     "show_progress" => true,
#     "iterations" => 100,
#     "n_samples" => 1000,
#     "n_warmup" => 100,
#     "n_chains" => 1,
#     "sampler" => "CmdStan",
#     "betas" => [0.25, 0.5, 0.75],
#     "beta_weights" => [1.25, 2.5],
#     "calibrate_beta_weight" => false,
#     "weights" => [0.0, 0.5, 1.0],
#     "metrics" => ["auc", "ll", "bf", "param_mse"],
#     "model_names" => ["beta", "weighted"],
#     "real_alphas" => [0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
#     "synth_alphas" => [0.0, 0.05, 0.1, 0.5, 1.0],
#     "synth_alpha_range" => [],
#     "real_alpha_range" => [],
#     "mu_p" => 1.0,
#     "algorithm" => "basic",
#     "seed" => 1,
#     "id" => "inline_test",
#     "sigma_p" => 50.0,
#     "alpha_p" => 2.0,
#     "beta_p" => 4.0,
#     "nu_p" => -1,
#     "Sigma_p" => -1,
#     "path" => ".",
#     "folds" => 5,
#     "split" => 1.0,
#     "use_ad" => false,
#     "distributed" => false,
#     "target_acceptance" => 0.8,
#     "mu" => 0.0,
#     "sigma" => 2.0,
#     "num_repeats" => 10,
#     "dataset" => "framingham",
#     "epsilon" => "6.0",
#     "label" => "TenYearCHD",
#     "folds" => 5,
#     "alphas" => []
# )

# @everywhere Turing.turnprogress(false)
# args = Dict(
#     "experiment_type" => "gaussian",
#     "show_progress" => true,
#     "iterations" => 100,
#     "n_samples" => 5000,
#     "n_warmup" => 1000,
#     "n_chains" => 1,
#     "sampler" => "Turing",
#     "scales" => [0.00001, 0.75],
#     "betas" => [0.25, 0.5, 0.75],
#     "beta_weights" => [1.25, 2.5],
#     "calibrate_beta_weight" => false,
#     "weights" => [],
#     "alphas" => [65, 70, 75, 80, 90, 100],
#     "metrics" => ["ll", "kld", "wass"],
#     "model_names" => ["weighted"],
#     "real_ns" => [15],
#     "real_n_range" => [],
#     "synth_ns" => [0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 250, 400, 600, 1000],
#     "synth_n_range" => [],
#     "n_unseen" => 500,
#     "algorithm" => "basic",
#     "mu_p" => 3.0,
#     "sigma_p" => 30.0,
#     "alpha_p" => 2.0,
#     "beta_p" => 4.0,
#     "nu_p" => -1,
#     "Sigma_p" => -1,
#     "path" => ".",
#     "folds" => 5,
#     "split" => 1.0,
#     "use_ad" => false,
#     "distributed" => false,
#     "target_acceptance" => 0.8,
#     "mu" => 0.0,
#     "sigma" => 2.0,
#     "num_repeats" => 10,
#     "id" => "fn15",
#     "seed" => 1,
#     "fn" => 15
# )

function main()

    args = parse_cl()
    experiment_type, base_path, metrics, algorithm, base_seed, id = (
        args["experiment_type"],
        args["path"],
        args["metrics"],
        args["algorithm"],
        args["seed"],
        args["id"]
    )
    αs = args["alphas"]
    Fₙ = args["fn"]
    dataset = (
        name = args["dataset"],
        label = args["label"],
        ϵ = args["epsilon"]
    )
    iterations, folds, split = (
        [i for i in 1:args["iterations"]],
        args["folds"],
        args["split"]
    )
    use_ad, distributed, show_progress = (
        args["use_ad"],
        args["distributed"],
        args["show_progress"]
    )
    mcmc = (
        sampler = args["sampler"],
        n_samples = args["n_samples"],
        n_warmup = args["n_warmup"],
        n_chains = args["n_chains"],
        target_acceptance = args["target_acceptance"],
        model_names = args["model_names"]
    )
    βs, βws, ws, calibrate_βw = (
        args["betas"],
        args["beta_weights"],
        args["weights"],
        args["calibrate_beta_weight"]
    )
    prior = (
        μₚ = args["mu_p"],
        σₚ = args["sigma_p"],
        αₚ = args["alpha_p"],
        βₚ = args["beta_p"],
        νₚ = args["nu_p"],
        Σₚ = args["Sigma_p"]
    )
    dgp = (
        μ = args["mu"],
        σ = args["sigma"]
    )
    sens = 6 * dgp[:σ]
    config = config_dict(experiment_type, args)

    Random.seed!(base_seed)
    @show args

    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    if distributed
        out_path = "$(base_path)/$(experiment_type)/outputs/$(id)_$(ENV["SLURM_JOB_ID"])_$(t)_$(mcmc[:sampler])"
    else
        out_path = "$(base_path)/$(experiment_type)/outputs/$(id)_$(t)_$(mcmc[:sampler])"
    end
    # plot_path = "$(base_path)/$(experiment_type)/plots/IN_RUN_$(t)_$(mcmc[:sampler])"
    mkpath(out_path)
    # mkpath(plot_path)

    # Pre-load and compile Stan models
    if mcmc[:sampler] in ["CmdStan", "Stan"]
        # mkpath("$(base_path)/$(experiment_type)/tmp$(mcmc[:sampler])/")
        stan_models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(base_path, experiment_type, mcmc...; dist = distributed))
            end
        )
    end

    # Load or create datasets
    if experiment_type == "gaussian"
        println("Generating data...")
        dgp = Distributions.Normal(dgp[:μ], dgp[:σ])
        if Fₙ > 0
            empirical_dgp = rand(dgp, Fₙ)
            all_synth_data_pre_noise = rand(empirical_dgp, (maximum(iterations), maximum(config[:synth_ns])))
        else
            all_synth_data_pre_noise = rand(dgp, (maximum(iterations), maximum(config[:synth_ns])))
        end
        all_real_data = rand(dgp, (maximum(iterations), maximum(config[:real_ns])))
        all_unseen_data = rand(dgp, (maximum(iterations), config[:n_unseen]))
    else
        labels, unshuffled_real_data, unshuffled_synth_data = load_data(dataset...)
        
        if experiment_type == "logistic_regression"
            θ_dim = size(unshuffled_real_data)[2] - 1
            unshuffled_real_data = standardise_out(unshuffled_real_data)
            unshuffled_synth_data = standardise_out(unshuffled_synth_data)
            unshuffled_real_data[:, labels[1]] = (2 .* unshuffled_real_data[:, labels[1]]) .- 1
            unshuffled_synth_data[:, labels[1]] = (2 .* unshuffled_synth_data[:, labels[1]]) .- 1
            data_levels = classes(categorical(unshuffled_real_data[:, labels[1]])[1])
        elseif experiment_type == "regression"
            θ_dim = length(predictors) + 1
            unshuffled_real_data[!, :course] = (unshuffled_real_data[!, :course] .- mean(unshuffled_real_data[!, :course])) / std(unshuffled_real_data[!, :course])
            unshuffled_synth_data[!, :course] = (unshuffled_synth_data[!, :course] .- mean(unshuffled_synth_data[!, :course])) / std(unshuffled_synth_data[!, :course])
        end
    end

    # βw calibration
    @show length(βws)
    βw_default = 1.25
    if calibrate_βw
        # βws = weight_calib(
        #     experiment_type,
        #     βw_default,
        #     Matrix(unshuffled_synth_data[:, Not(labels)]),
        #     Int.(unshuffled_synth_data[:, labels[1]]),
        #     β, λ
        # )
        @show "TODO"
    elseif length(βws) < 1
        βws = [βw_default]
    end
    @show βws

    # Work out the baseline best case model parameters to compare against using all of the real data
    if experiment_type == "logistic_regression"
        lr = LogisticRegression(1.0, fit_intercept = false)
        θ_real = MLJLinearModels.fit(
            lr,
            Matrix(unshuffled_real_data[:, Not(dataset[:label])]),
            Array(unshuffled_real_data[:, dataset[:label]]);
            solver = MLJLinearModels.LBFGS()
        )
    end

    model_configs = generate_model_configs(mcmc[:model_names], βs, βws, ws, αs)

    # Instantiate the output file according to the passed MCMC config
    if "param_mse" in config[:metrics]
        name_metrics = join(vcat(config[:metrics], ["param_mse_$(p)" for p in names(unshuffled_real_data)]), ",")
    else
        name_metrics = join(config[:metrics], ",")
    end
    init_csv_files(experiment_type, distributed, out_path, name_metrics)

    S = generate_all_steps(experiment_type, algorithm, iterations, config, model_configs)
    # @showprogress for s in S
    p = Progress(size(S)[1])
    progress_pmap(1:size(S)[1], progress=p) do i

        s = S[i]

        if experiment_type == "gaussian"

            c = (
                i = s[1],
                real_n = s[2],
                synth_n = s[3],
                n_unseen = s[4],
                λ = s[5],
                model = s[6][:model] == "resampled" ? "weighted" : s[6][:model],
                w = length(αs) > 0 ? s[6][:weight] / s[3] : s[6][:weight],
                β = s[6][:β],
                metric = s[end]
            )
            @show s

            Random.seed!(base_seed + c[:i])

            real_data = all_real_data[c[:i], 1:c[:real_n]]
            if s[6][:model] == "resampled"
                synth_data = rand(real_data, c[:synth_n])
            else
                synth_data = (
                    all_synth_data_pre_noise[c[:i], :] +
                    rand(Laplace(0, c[:λ]), maximum(config[:synth_ns]))
                )[1:c[:synth_n]]
            end
            unseen_data = all_unseen_data[c[:i], :]
            
            if config[:algorithm] == "basic"

                if mcmc[:sampler] == "CmdStan"
    
                    data = Dict(
                        "n" => c[:real_n],
                        "y1" => real_data,
                        "m" => length(synth_data),
                        "y2" => synth_data,
                        "p_mu" => prior[:μₚ],
                        "p_alpha" => prior[:αₚ],
                        "p_beta" => prior[:βₚ],
                        "hp" => prior[:σₚ],
                        "scale" => c[:λ],
                        "beta" => c[:β] == c[:β],
                        "beta_w" => c[:w] == c[:w],
                        "w" => c[:w] == c[:w],
                        "lambda" => c[:λ]
                    )
                    model = models[myid()]["$(c[:model])_$(myid())"]
    
                    rc, chn, _ = stan(
                        model,
                        data;
                        summary=false
                    )
                    if rc == 0
                        chains = Array(chn)[:, 1:2]
                        ms = evaluate_samples(unseen_data, dgp, chains)
                    end
    
                elseif mcmc[:sampler] == "AHMC"
    
                    # Define log posteriors and gradients of them
                    b, models = init_ahmc_gaussian_models(
                        real_data, synth_data, c[:w], c[:w], c[:β], c[:λ], prior[:αₚ], prior[:βₚ], prior[:μₚ], prior[:σₚ]
                    )
                    model = models[c[:model]]

                    chains = map(1:mcmc[:n_chains]) do i
                        m = DiagEuclideanMetric(2)
                        initial_θ = rand(2)
                        hamiltonian, proposal, adaptor = setup_run(
                            model,
                            m,
                            initial_θ;
                            target_acceptance_rate = mcmc[:target_acceptance]
                        )
                        chn, _ = sample(
                            hamiltonian, proposal, initial_θ, mcmc[:n_samples], adaptor, mcmc[:n_warmup];
                            drop_warmup=true, progress=show_progress, verbose=show_progress
                        )
                        chn
                    end
                    chains = hcat(b.(vcat(chains...))...)'
                    # @show mean(chains, dims=1)

                    ms = evaluate_gaussian_samples(unseen_data, dgp, chains)
    
                elseif mcmc[:sampler] == "Turing"
    
                    models = init_turing_gaussian_models(
                        real_data, synth_data, c[:w], c[:w], c[:β], c[:λ], prior[:αₚ], prior[:βₚ], prior[:μₚ], prior[:σₚ]
                    )
                    model = models[c[:model]]
    
                    chains = map(1:mcmc[:n_chains]) do i
                        chn = sample(model, Turing.NUTS(mcmc[:n_warmup], mcmc[:target_acceptance]), mcmc[:n_samples])
                    end
                    ms = evaluate_gaussian_samples(unseen_data, dgp, Array(vcat(chains...)))
    
                end

                ll, kl, wass = ms["ll"], ms["kld"], ms["wass"]
    
                open("$(out_path)/$(myid())_out.csv", "a") do io
                    write(io, "$(base_seed+c[:i]),$(c[:i]),$(dgp.σ),$(dgp.μ),$(c[:λ]),$(s[:6][:model]),$(length(αs) > 0 ? s[6][:weight] : -1),$(c[:w]),$(c[:β]),$(c[:real_n]),$(length(synth_data)),$(ll),$(kl),$(wass),$(sens / c[:λ])\n")
                end

            else

                if c[:metric] == "ll"
                    gtol = 5
                else
                    gtol = 0.02
                end

                max_syn = maximum(real_ns) - real_n + min_synth_n
        
                function g(synth_data)
        
                    evaluations = []
                    if sampler == "CmdStan"
        
                        data = Dict(
                            "n" => real_n,
                            "y1" => real_data,
                            "m" => length(synth_data),
                            "y2" => synth_data,
                            "p_mu" => μₚ,
                            "p_alpha" => αₚ,
                            "p_beta" => βₚ,
                            "hp" => σₚ,
                            "scale" => λ,
                            "beta" => β,
                            "beta_w" => βw,
                            "w" => w,
                            "lambda" => λ
                        )
                        model = models[myid()]["$(model_name)_$(myid())"]
        
                        rc, chn, _ = stan(
                            model,
                            data;
                            summary=false
                        )
                        if rc == 0
                            chains = Array(chn)[:, 1:2]
                            # @show mean(chains, dims=1)
                            ms = evaluate_samples(unseen_data, dgp, chains)
                        end
        
                    elseif sampler == "AHMC"
        
                        # Define log posteriors and gradients of them
                        b, models = init_ahmc_models(
                            real_data, synth_data, w, βw, β, λ, αₚ, βₚ, μₚ, σₚ
                        )
                        model = models[model_name]
        
                        chains = map(1:nchains) do i
                            m = DiagEuclideanMetric(2)
                            initial_θ = rand(2)
                            hamiltonian, proposal, adaptor = setup_run(
                                model,
                                m,
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
                        # @show mean(chains, dims=1)
                        ms = evaluate_samples(unseen_data, dgp, chains)
        
                    elseif sampler == "Turing"
        
                        models = init_turing_models(
                            real_data, synth_data, w, βw, β, λ, αₚ, βₚ, μₚ, σₚ
                        )
                        model = models[model_name]
        
                        chains = map(1:nchains) do i
                            chn = sample(model, Turing.NUTS(n_warmup, target_acceptance_rate), n_samples)
                            Array(chn)
                        end
                        @show size(vcat(chains...))
                        @show mean(vcat(chains...), dims=1)
                        ms = evaluate_samples(unseen_data, dgp, vcat(chains...))
        
                    end
        
                    ll, kl, wass = ms["ll"], ms["kld"], ms["wass"]
        
                    open("$(out_path)/$(myid())_out.csv", "a") do io
                        write(io, "$(c[:i]),$(λ),$(model_name),$(real_n),$(length(synth_data)),$(ll),$(kl),$(wass)\n")
                    end
        
                    return ms[metric]
                end
        
                println("Worker $(myid()) on c[:i]: $(c[:i]), noise: $(λ), realn: $(real_n), model: $(model_name), metric: $(metric), max synthn: $(max_syn)...")
        
                # Assuming unimodality we can safely step forward in xmax until we see an increase in g,
                # then run a search algorithm ideally within xprev2 and xmax to find the turning point
                xmin, xmax = 0, min_synth_n
                synths = [gen_synth(max_syn, dgp, λ) for _ in 1:config[:K]]
                gmins = [g(synths[i][1:xmin]) for i in 1:config[:K]]
                gmaxs = [g(synths[i][1:xmax]) for i in 1:config[:K]]
                p = plot([xmin, xmax], [mean(gmins), mean(gmaxs)])
                display(p)
                xprev, gprevs, xprev2, gprevs2, xprev3 = xmin, gmins, xmin, gmins, xmin
                while (mean(gmaxs) < mean(gprevs)) & (xmax * 2 <= max_syn)
                    xprev3 = xprev2
                    xprev2, gprevs2 = xprev, gprevs
                    xprev, gprevs = xmax, gmaxs
                    xmax *= 2
                    gmaxs = [g(synths[i][1:xmax]) for i in 1:config[:K]]
                    plot!([xprev2, xprev, xmax], [mean(gprevs2), mean(gprevs), mean(gmaxs)])
                    display(p)
                end
        
                if mean(gmaxs) < mean(gprevs)
        
                    png(p, "$(plot_path)/DONE_xs_iter$(c[:i])noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")
                    xmin, xmid₁, xmid₂ = xprev3, xprev2, xprev
        
                else
        
                    png(p, "$(plot_path)/INIT_xs_iter$(c[:i])noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")
        
                    # Check if xprev2 was updated, i.e. enough iters in prev step, to set xmin to something above 0
                    if xprev != xprev2
                        xmin = xprev2
                        gmins = gprevs2
                    end
        
                    if alg == "golden"
        
                        h = xmax - xmin
                        N = ceil(Int, log(1 / h) / log(invϕ))
                        xmid₁ = round(Int, xmin + invϕ² * h)
                        xmid₂ = round(Int, xmin + invϕ * h)
        
                        gmid₁s = [g(synths[i][1:xmid₁]) for i in 1:config[:K]]
                        gmid₂s = [g(synths[i][1:xmid₂]) for i in 1:config[:K]]
                        gdiffs = gmid₂s - gmid₁s
                        ci = confint(OneSampleTTest(gdiffs))
        
                        p = plot([xmin, xmid₁, xmid₂, xmax], [mean(gmins), mean(gmid₁s), mean(gmid₂s), mean(gmaxs)])
        
                        for n in 1:N
        
                            # display(p)
        
                            if (xmid₁ == xmid₂ == xmin) | (xmid₁ == xmid₂ == xmax)
                                break
                            elseif (abs(mean(gmid₁s) - mean(gmins)) < gtol) &
                                (abs(mean(gdiffs)) < gtol) &
                                (abs(mean(gmid₂s) - mean(gmaxs)) < gtol) &
                                (abs(mean(gmid₂s) - mean(gmins)) < gtol) &
                                (abs(mean(gmid₁s) - mean(gmaxs)) < gtol) &
                                (abs(mean(gmins) - mean(gmaxs)) < gtol)
                                @show "Breaking due to gdiffall"
                                break
                            end
        
                            synths = [gen_synth(xmax, dgp, λ) for _ in 1:config[:K]]
        
                            if mean(ci) > 0
                                xmax = xmid₂
                                xmid₂ = xmid₁
                                h = invϕ * h
                                xmid₁ = round(Int, xmin + invϕ² * h)
                                gmaxs = gmid₂s
                                gmid₂s = gmid₁s
                                gmid₁s = [g(synths[i][1:xmid₁]) for i in 1:config[:K]]
                            else
                                xmin = xmid₁
                                xmid₁ = xmid₂
                                h = invϕ * h
                                xmid₂ = round(Int, xmin + invϕ * h)
                                gmins = gmid₁s
                                gmid₁s = gmid₂s
                                gmid₂s = [g(synths[i][1:xmid₂]) for i in 1:config[:K]]
                            end
        
                            gdiffs = gmid₂s - gmid₁s
                            ci = confint(OneSampleTTest(gdiffs))
        
                            plot!([xmin, xmid₁, xmid₂, xmax], [mean(gmins), mean(gmid₁s), mean(gmid₂s), mean(gmaxs)])
        
                        end
        
                        png(p, "$(plot_path)/DONE_xs_iter$(c[:i])noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")
        
                    else
        
                        synth_δ = 5
                        ks = DiscreteNonParametric(collect(xmin:xmax), [1 / (xmax - xmin + 1) for _ in xmin:xmax])
        
                        while maximum(ks.p) < maximum(10 / (xmax - xmin), 0.12)
        
                            synth_n = rand(ks)
                            pos = 0
                            for k in 1:config[:K]
        
                                synth_data_a = rand(dgp, Int(synth_n + synth_δ)) + rand(Laplace(0, λ), Int(synth_n + synth_δ))
                                synth_data_b = synth_data_a[1:synth_n]
                                difference = g(synth_data_a) - g(synth_data_b)
                                if difference > 0
                                    pos += 1
                                end
        
                            end
        
                            @show synth_n
                            @show pos
                            p̂ = cdf(
                                Beta(0.125, 0.125),
                                maximum([(pos + 1) / (config[:K] + 2), 1 - (pos + 1) / (config[:K] + 2)])
                            )
                            @show p̂
                            l_mult = p̂ ^ (pos + 1) * (1 - p̂) ^ (config[:K] + 2 - pos)
                            @show l_mult
                            r_mult = (1 - p̂) ^ (pos + 1) * p̂ ^ (config[:K] + 2 - pos)
                            @show r_mult
                            new_ps = vcat(
                                ks.p[1:(synth_n - xmin + 1)] .* l_mult,
                                ks.p[(synth_n - xmin + 2):end] .* r_mult
                            )
                            ks = DiscreteNonParametric(collect(xmin:xmax), new_ps ./ sum(new_ps))
                            p = plot(xmin:xmax, ks.p)
                            display(p)
        
                        end
        
                        xmin = argmax(ks.p)
                        xmax = length(ks.p) - argmax(reverse(ks.p)) + 1
                        h = xmax - xmin
                        xmid₁ = round(Int, xmin + invϕ² * h)
                        xmid₂ = round(Int, xmin + invϕ * h)
        
                        png(p, "$(plot_path)/DONE_xs_iter$(c[:i])noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")
        
                    end
        
                end

                open("$(out_path)/$(myid())_tps.csv", "a") do io
                    write(io, "$(c[:i]),$(λ),$(real_n),$(model_name),$(metric),$(xmin),$(xmid₁),$(xmid₂),$(xmax)\n")
                end

            end

        elseif experiment_type == "logistic_regression"

            c = (
                i = s[1],
                real_α = s[2],
                synth_α = s[3],
                fold = s[4],
                model = s[5][:model] == "resampled" ? "weighted" : s[5][:model],
                w = s[5][:weight],
                β = s[5][:β]
            )
            @show c

            Random.seed!(base_seed + c[:i])
            real_data = unshuffled_real_data[shuffle(axes(unshuffled_real_data, 1)), :]
            synth_data = unshuffled_synth_data[shuffle(axes(unshuffled_synth_data, 1)), :]

            X_real, y_real, X_synth, y_synth, X_valid, y_valid = fold_α(
                real_data, synth_data, c[:real_α], c[:synth_α],
                c[:fold], config[:folds], labels
            )

            if s[5][:model] == "resampled"
                idx = rand(collect(1:size(X_real)[1]), length(y_synth))
                X_synth = X_real[idx, :]
                y_synth = y_real[idx]
            end

            # First parameter here is regularisation on the LogisticRegression, choose what to do with it
            initial_θ = init_run(
                1.0, X_real, y_real
            )

            if mcmc[:sampler] == "CmdStan"

                data = Dict(
                    "f" => θ_dim - 1,
                    "a" => size(X_real)[1] == 0 ? 1 : size(X_real)[1],
                    "X_real" => size(X_real)[1] == 0 ? zeros(1, size(X_real)[2])[:, 2:end] : X_real[:, 2:end],
                    "y_real" => size(X_real)[1] == 0 ? [1] : Int.((y_real .+ 1) ./ 2),
                    "b" => size(X_synth)[1] == 0 ? 1 : size(X_synth)[1],
                    "X_synth" => size(X_synth)[1] == 0 ? zeros(1, size(X_synth)[2])[:, 2:end] : X_synth[:, 2:end],
                    "y_synth" => size(X_synth)[1] == 0 ? [1] : Int.((y_synth .+ 1) ./ 2),
                    "w" => c[:w],
                    "beta" => c[:β],
                    "beta_w" => c[:w],
                    "flag_real" => size(X_real)[1] == 0 ? 1 : 0,
                    "flag_synth" => size(X_synth)[1] == 0 ? 1 : 0
                )
                # init = Dict(
                #     "alpha" => initial_θ[1],
                #     "coefs" => initial_θ[2:end]
                # )
                init = Dict(
                    "alpha" => 0.0,
                    "coefs" => zeros(θ_dim - 1)
                )
                model = stan_models[myid()]["$(c[:model])_$(myid())"]

                try
                    @time _, chn, _ = stan(
                        model,
                        data;
                        init=init
                    )
                    chains = Array(chn)[:, 1:θ_dim]
                    auc_score, ll, bf, param_mse, param_mses = evaluate_logistic_samples(X_valid, y_valid, chains, data_levels, θ_real)
                catch
                    auc_score, ll, bf, param_mse, param_mses = NaN, NaN, NaN, NaN, [NaN for i in 1:θ_dim]
                end
    
            elseif mcmc[:sampler] == "Stan"

                # Unsure if this works but want to eventually transition to using Stan
                data = Dict(
                    "f" => θ_dim - 1,
                    "a" => size(X_real)[1] == 0 ? 1 : size(X_real)[1],
                    "X_real" => size(X_real)[1] == 0 ? zeros(1, size(X_real)[2])[:, 2:end] : X_real[:, 2:end],
                    "y_real" => size(X_real)[1] == 0 ? [1] : Int.((y_real .+ 1) ./ 2),
                    "b" => size(X_synth)[1] == 0 ? 1 : size(X_synth)[1],
                    "X_synth" => size(X_synth)[1] == 0 ? zeros(1, size(X_synth)[2])[:, 2:end] : X_synth[:, 2:end],
                    "y_synth" => size(X_synth)[1] == 0 ? [1] : Int.((y_synth .+ 1) ./ 2),
                    "w" => c[:w],
                    "beta" => c[:β],
                    "beta_w" => c[:w],
                    "flag_real" => size(X_real)[1] == 0 ? 1 : 0,
                    "flag_synth" => size(X_synth)[1] == 0 ? 1 : 0
                )
                init = Dict(
                    "alpha" => initial_θ[1],
                    "coefs" => initial_θ[2:end]
                )
                model = stan_models[myid()]["$(c[:model])_$(myid())"]

                rc = stan_sample(
                    model;
                    init=init,
                    data=data,
                )
                if success(rc)
                    samples = mean(read_samples(model)[:, 1:θ_dim, :], dims=3)[:, :, 1]
                    auc_score, ll, bf, param_mse, param_mses = evaluate_logistic_samples(X_valid, y_valid, samples, c)
                else
                    auc_score, ll, bf, param_mse, param_mses = NaN, NaN, NaN, NaN, [NaN for i in 1:θ_dim]
                end
    
            elseif mcmc[:sampler] == "AHMC"
    
                # Define log posteriors and gradients of them
                ahmc_models = init_ahmc_logistic_models(
                    X_real, y_real, X_synth, y_synth, prior[:σₚ], c[:w], c[:w], c[:β], initial_θ
                )
                model = ahmc_models[c[:model]]

                chains = map(1:mcmc[:n_chains]) do i
                    metric = DiagEuclideanMetric(θ_dim)
                    hamiltonian, proposal, adaptor = setup_run(
                        model.ℓπ,
                        metric,
                        initial_θ;
                        ∂ℓπ∂θ = model.∇ℓπ,
                        target_acceptance_rate = mcmc[:target_acceptance]
                    )
                    chn, _ = sample(
                        hamiltonian, proposal, initial_θ, mcmc[:n_samples], adaptor, mcmc[:n_warmup];
                        drop_warmup=true, progress=show_progress, verbose=show_progress
                    )
                    chn
                end
        
                chains = hcat(vcat(chains...)...)'
                auc_score, ll, bf, param_mse, param_mses = evaluate_logistic_samples(X_valid, y_valid, chains, data_levels, θ_real)
    
            elseif mcmc[:sampler] == "Turing"
    
                turing_models = init_turing_logistic_models(
                    X_real, y_real, X_synth, y_synth, prior[:σₚ], c[:w], c[:w], c[:β]
                )
                model = turing_models[c[:model]]
        
                @time chains = map(1:mcmc[:n_chains]) do i
                    varinfo = Turing.VarInfo(model)
                    model(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((θ = initial_θ,)))
                    init_θ = varinfo[Turing.SampleFromPrior()]
                    chn = sample(model, Turing.NUTS(mcmc[:n_warmup], mcmc[:target_acceptance]), mcmc[:n_samples], init_theta = init_θ)
                    chn
                end

                chains = Array(vcat(chains...))
                auc_score, ll, bf, param_mse, param_mses = evaluate_logistic_samples(X_valid, y_valid, chains, data_levels, θ_real)
        
            end
    
            open("$(out_path)/$(myid())_out.csv", "a") do io
                write(io, """$(base_seed+c[:i]),$(c[:i]),$(c[:fold]),$(dataset[:name]),$(dataset[:label]),$(dataset[:ϵ]),$(s[5][:model]),$(c[:w]),$(c[:β]),$(c[:real_α]),$(c[:synth_α]),$(auc_score),$(ll),$(bf),$(param_mse),$(join(param_mses,","))\n""")
            end

        elseif experiment_type == "regression"
            # TODO
            print("Not implemented")
        end

    end

end

main()
