using ClusterManagers
using Distributed
# addprocs(4)
# @everywhere begin
#     using Pkg; Pkg.activate("."); Pkg.instantiate()
# end

# addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])), o=string(ENV["SLURM_JOB_ID"]))
addprocs_slurm(parse(Int, ENV["SLURM_NTASKS"]))
println("We are all connected and ready.")
for i in workers()
    host, pid = fetch(@spawnat i (gethostname(), getpid()))
    println(host, pid)
end
# using Pkg; Pkg.activate("/home/dcs/csrxgb/synthetic/Project.toml")
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
using StatsFuns: invsqrt2π, log2π, sqrt2
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
    using StatsFuns: invsqrt2π, log2π, sqrt2
    using Turing
    using Zygote
    include("src/distributions.jl")
    include("src/evaluation.jl")
    include("src/init.jl")
    include("src/loss.jl")
    include("src/utils.jl")
    include("src/weight_calibration.jl")
end

# args = Dict(
#     "experiment_type" => "gaussian",
#     "show_progress" => true,
#     "iterations" => 100,
#     "n_samples" => 10000,
#     "n_warmup" => 1000,
#     "n_chains" => 1,
#     "sampler" => "Turing",
#     "scales" => [0.5, 1.5, 2.5],
#     "betas" => [0.25, 0.5, 0.75],
#     "beta_weights" => [1.25, 2.5, 4],
#     "calibrate_beta_weight" => false,
#     "weights" => [0.0, 0.25, 0.5, 0.75, 1.0],
#     "metrics" => ["ll", "kld", "wass"],
#     "model_names" => ["beta", "weighted", "noise_aware"],
#     "real_ns" => [10, 25, 50, 100],
#     "synth_ns" => [5, 10, 15, 20, 30, 40, 50],
#     "n_unseen" => 100,
#     "algorithm" => "basic",
#     "mu_p" => 1.0,
#     "sigma_p" => 1.0,
#     "alpha_p" => 2.0,
#     "beta_p" => 4.0,
#     "nu_p" => -1,
#     "Sigma_p" => -1,
#     "path" => ".",
#     "folds" => 5,
#     "split" => 1.0,
#     "use_ad" => false,
#     "distributed" => true,
#     "no_shuffle" => true,
#     "target_acceptance" => 0.8,
#     "mu" => 0.0,
#     "sigma" => 2.0,
#     "num_repeats" => 10
# )

function main()

    args = parse_cl()
    experiment_type, path, metrics, algorithm, base_seed, id = (
        args["experiment_type"],
        args["path"],
        args["metrics"],
        args["algorithm"],
        args["seed"],
        args["id"]
    )
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
    use_ad, distributed, no_shuffle, show_progress = (
        args["use_ad"],
        args["distributed"],
        args["no_shuffle"],
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
    config = config_dict(experiment_type, args)
    Random.seed!(base_seed)
    @show args

    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    out_path = "$(path)/$(experiment_type)/outputs/$(id)_$(t)_$(mcmc[:sampler])"
    # plot_path = "$(path)/$(experiment_type)/plots/IN_RUN_$(t)_$(mcmc[:sampler])"
    mkpath(out_path)
    # mkpath(plot_path)

    # Pre-load and compile Stan models
    if mcmc[:sampler] in ["CmdStan", "Stan"]
        mkpath("$(path)/$(experiment_type)/tmp$(mcmc[:sampler])/")
        models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(path, experiment_type, mcmc...; dist = distributed))
            end
        )
    end

    # Load or create datasets
    if experiment_type == "gaussian"
        println("Generating data...")
        dgp = Distributions.Normal(dgp[:μ], dgp[:σ])
        all_real_data = rand(dgp, (maximum(iterations), maximum(config[:real_ns])))
        all_synth_data_pre_noise = rand(dgp, (maximum(iterations), maximum(config[:synth_ns])))
        all_unseen_data = rand(dgp, (maximum(iterations), config[:n_unseen]))
    else
        labels, unshuffled_real_data, unshuffled_synth_data = load_data(dataset...)
        
        if experiment_type == "logistic_regression"
            θ_dim = size(unshuffled_real_data)[2] - 1
            unshuffled_real_data[:, labels[1]] = (2 .* unshuffled_real_data[:, labels[1]]) .- 1
            unshuffled_synth_data[:, labels[1]] = (2 .* unshuffled_synth_data[:, labels[1]]) .- 1
            c = classes(categorical(unshuffled_real_data[:, labels[1]])[1])
        elseif experiment_type == "regression"
            θ_dim = length(predictors) + 1
            unshuffled_real_data[!, :course] = (unshuffled_real_data[!, :course] .- mean(unshuffled_real_data[!, :course])) / std(unshuffled_real_data[!, :course])
            unshuffled_synth_data[!, :course] = (unshuffled_synth_data[!, :course] .- mean(unshuffled_synth_data[!, :course])) / std(unshuffled_synth_data[!, :course])
        end
    end

    # βw calibration
    @show length(βws)
    βw_default = 1.2
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

    model_configs = generate_model_configs(mcmc[:model_names], βs, βws, ws)

    # Instantiate the output file aaccording to the passed MCMC config
    name_metrics = join(config[:metrics], ",")
    if distributed
        @everywhere begin
            open("$($out_path)/$(myid())_out.csv", "w") do io
                write(io, "seed,iter,noise,model,weight,beta,real_n,synth_n,$($name_metrics)\n")
            end
        end
    else
        begin
            open("$(out_path)/$(myid())_out.csv", "w") do io
                write(io, "seed,iter,noise,model,weight,beta,real_n,synth_n,$(name_metrics)\n")
            end
        end
    end

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
                model = s[6][:model],
                w = s[6][:weight],
                β = s[6][:β],
                metric = s[end]
            )
            @show c

            Random.seed!(base_seed + c[:i])
    
            real_data = all_real_data[c[:i], 1:c[:real_n]]
            synth_data = (
                all_synth_data_pre_noise[c[:i],:] +
                rand(Laplace(0, c[:λ]), maximum(config[:synth_ns]))
            )[1:c[:synth_n]]
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
                    @time b, models = init_ahmc_gaussian_models(
                        real_data, synth_data, c[:w], c[:w], c[:β], c[:λ], prior[:αₚ], prior[:βₚ], prior[:μₚ], prior[:σₚ]
                    )
                    model = models[c[:model]]
    
                    @time chains = map(1:mcmc[:n_chains]) do i
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
                    @time chains = hcat(b.(vcat(chains...))...)'
                    # @show mean(chains, dims=1)
                    @time ms = evaluate_gaussian_samples(unseen_data, dgp, chains)
    
                elseif mcmc[:sampler] == "Turing"
    
                    @time models = init_turing_gaussian_models(
                        real_data, synth_data, c[:w], c[:w], c[:β], c[:λ], prior[:αₚ], prior[:βₚ], prior[:μₚ], prior[:σₚ]
                    )
                    model = models[c[:model]]
    
                    @time chains = map(1:mcmc[:n_chains]) do i
                        chn = sample(model, Turing.NUTS(mcmc[:n_warmup], mcmc[:target_acceptance]), mcmc[:n_samples])
                    end
                    @time ms = evaluate_gaussian_samples(unseen_data, dgp, Array(vcat(chains...)))
    
                end
    
                ll, kl, wass = ms["ll"], ms["kld"], ms["wass"]
    
                @time open("$(out_path)/$(myid())_out.csv", "a") do io
                    write(io, "$(base_seed+c[:i]),$(c[:i]),$(c[:λ]),$(c[:model]),$(c[:w]),$(c[:β]),$(c[:real_n]),$(length(synth_data)),$(ll),$(kl),$(wass)\n")
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
            # TODO
            print("Not implemented")
        elseif experiment_type == "regression"
            # TODO
            print("Not implemented")
        end

    end

end

main()
