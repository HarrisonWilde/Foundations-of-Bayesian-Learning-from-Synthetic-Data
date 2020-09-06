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
using DataStructures
using Statistics
using HypothesisTests
using Plots
using Plots
# include("../common/utils.jl")
# include("../common/init.jl")
# include("distributions.jl")
# include("loss.jl")
# include("evaluation.jl")
# include("init.jl")
# include("weight_calibration.jl")

@everywhere begin
    using Distributed
    using Bijectors
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
    using DataStructures
    using Statistics
    using HypothesisTests
    using Plots
    # include("src/common/utils.jl")
    # include("src/common/init.jl")
    # include("src/gaussian/init.jl")
    # include("src/gaussian/distributions.jl")
    # include("src/gaussian/evaluation.jl")
    # include("src/gaussian/loss.jl")
    # include("src/gaussian/weight_calibration.jl")
end

# λs, K = [1.0, 1.25, 1.5, 1.75, 2.0], 100
# path, iterations = ".", 5
# distributed, sampler, no_shuffle, alg = false, "AHMC", false, "bisection"

function run_experiment()


    args = parse_cl()
    λs, K, algorithm = args["scales"], args["num_repeats"], args["algorithm"]
    path, dataset, label, ε = args["path"], args["dataset"], args["label"], args["epsilon"]
    iterations, folds, split = args["iterations"], args["folds"], args["split"]
    use_ad, distributed, sampler, no_shuffle = args["use_ad"], args["distributed"], args["sampler"], args["no_shuffle"]
    # experiment_type = args["experiment"]
    experiment_type = "gaussian"

    t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
    out_path = "$(path)/src/$(experiment_type)/outputs/$(t)_$(sampler)"
    plot_path = "$(path)/src/$(experiment_type)/plots/IN_RUN_$(t)_$(sampler)"
    mkpath(out_path)
    mkpath(plot_path)

    println("Setting up experiment...")
    w = 0.5
    β = 0.5
    βw = 1.25
    μ = 0.
    σ = 1.
    αₚ, βₚ, μₚ, σₚ = 3., 5., 1., 1.
    # real_ns = vcat([1, 5, 10, 50, 100, 150, 250, 400], collect(1:6) .^ 2 .* 11)
    # real_ns = [1, 5, 10, 50, 100, 150, 250, 400]
    real_ns = [1]
    metrics = [
        "ll",
        "kld",
        "wass"
    ]
    model_names = [
        "beta",
        "weighted",
        "naive",
        "no_synth",
        "beta_all",
        "noise_aware"
    ]
    min_synth_n = 25
    unseen_n = 1000

    num_real_ns = length(real_ns)
    num_λs = length(λs)
    num_metrics = length(metrics)
    num_models = length(model_names)

    model_steps = num_metrics
    real_n_steps = model_steps * num_models
    λ_steps = real_n_steps * num_real_ns
    iter_steps = λ_steps * num_λs
    total_steps = iter_steps * iterations

    n_samples, n_warmup = 12500, 2500
    nchains = 1
    target_acceptance_rate = 0.8
    if (sampler == "CmdStan") | (sampler == "Stan")
        mkpath("$(path)/src/$(experiment_type)/tmp$(sampler)/")
        models = Dict(
            pmap(1:nworkers()) do i
                (myid() => init_stan_models(path, sampler, model_names, experiment_type, target_acceptance_rate, nchains, n_samples, n_warmup; dist = distributed))
            end
        )
    end
    show_progress = distributed

    metrics_string = join([metric for metric in metrics], ",")
    @everywhere begin
        open("$($out_path)/$(myid())_out.csv", "w") do io
            write(io, "iter,scale,model_name,real_n,synth_n,$($metrics_string)\n")
        end
        open("$($out_path)/$(myid())_tps.csv", "w") do io
            write(io, "iter,scale,real_n,model_name,metric,xmin,xmid1,xmid2,xmax\n")
        end
    end

    println("Generating data...")
    dgp = Distributions.Normal(μ, σ)
    all_real_data = rand(dgp, (iterations, maximum(real_ns)))
    all_unseen_data = rand(dgp, (iterations, unseen_n))
    println("Distributing work...")
    p = Progress(total_steps)

    βws = zeros(0)
    for λ in λs
        βw_calib = weight_calib(
            rand(dgp, maximum(real_ns)) + rand(Laplace(0, λ), maximum(real_ns)),
            β, αₚ, βₚ, μₚ, σₚ
        )
        if isnan(βw_calib)
            βw_calib = βw
        elseif βw_calib > 5
            βw_calib = 5.
        elseif βw_calib < 0.5
            βw_calib = 0.5
        end
        append!(βws, βw_calib)
    end
    @show βws

    progress_pmap(1:total_steps, progress=p) do i

        iter = ceil(Int, i / iter_steps)
        iterᵢ = ((i - 1) % iter_steps) + 1
        λ = λs[ceil(Int, iterᵢ / λ_steps)]
        βw = βws[ceil(Int, iterᵢ / λ_steps)]
        λᵢ = ((iterᵢ - 1) % λ_steps) + 1
        real_n = real_ns[ceil(Int, λᵢ / real_n_steps)]
        real_nᵢ = ((λᵢ - 1) % real_n_steps) + 1
        model_name = model_names[ceil(Int, real_nᵢ / model_steps)]
        modelᵢ = ((real_nᵢ - 1) % model_steps) + 1
        metric = metrics[modelᵢ]

        if metric == "ll"
            gtol = 5
        else
            gtol = 0.02
        end

        real_data = all_real_data[iter, 1:real_n]
        unseen_data = all_unseen_data[iter, :]
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
                write(io, "$(iter),$(λ),$(model_name),$(real_n),$(length(synth_data)),$(ll),$(kl),$(wass)\n")
            end

            return ms[metric]
        end

        println("Worker $(myid()) on iter: $(iter), noise: $(λ), realn: $(real_n), model: $(model_name), metric: $(metric), max synthn: $(max_syn)...")

        # Assuming unimodality we can safely step forward in xmax until we see an increase in g,
        # then run a search algorithm ideally within xprev2 and xmax to find the turning point
        xmin, xmax = 0, min_synth_n
        synths = [gen_synth(max_syn, dgp, λ) for _ in 1:K]
        gmins = [g(synths[i][1:xmin]) for i in 1:K]
        gmaxs = [g(synths[i][1:xmax]) for i in 1:K]
        p = plot([xmin, xmax], [mean(gmins), mean(gmaxs)])
        display(p)
        xprev, gprevs, xprev2, gprevs2, xprev3 = xmin, gmins, xmin, gmins, xmin
        while (mean(gmaxs) < mean(gprevs)) & (xmax * 2 <= max_syn)
            xprev3 = xprev2
            xprev2, gprevs2 = xprev, gprevs
            xprev, gprevs = xmax, gmaxs
            xmax *= 2
            gmaxs = [g(synths[i][1:xmax]) for i in 1:K]
            plot!([xprev2, xprev, xmax], [mean(gprevs2), mean(gprevs), mean(gmaxs)])
            display(p)
        end

        if mean(gmaxs) < mean(gprevs)

            png(p, "$(plot_path)/DONE_xs_iter$(iter)noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")
            xmin, xmid₁, xmid₂ = xprev3, xprev2, xprev

        else

            png(p, "$(plot_path)/INIT_xs_iter$(iter)noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")

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

                gmid₁s = [g(synths[i][1:xmid₁]) for i in 1:K]
                gmid₂s = [g(synths[i][1:xmid₂]) for i in 1:K]
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

                    synths = [gen_synth(xmax, dgp, λ) for _ in 1:K]

                    if mean(ci) > 0
                        xmax = xmid₂
                        xmid₂ = xmid₁
                        h = invϕ * h
                        xmid₁ = round(Int, xmin + invϕ² * h)
                        gmaxs = gmid₂s
                        gmid₂s = gmid₁s
                        gmid₁s = [g(synths[i][1:xmid₁]) for i in 1:K]
                    else
                        xmin = xmid₁
                        xmid₁ = xmid₂
                        h = invϕ * h
                        xmid₂ = round(Int, xmin + invϕ * h)
                        gmins = gmid₁s
                        gmid₁s = gmid₂s
                        gmid₂s = [g(synths[i][1:xmid₂]) for i in 1:K]
                    end

                    gdiffs = gmid₂s - gmid₁s
                    ci = confint(OneSampleTTest(gdiffs))

                    plot!([xmin, xmid₁, xmid₂, xmax], [mean(gmins), mean(gmid₁s), mean(gmid₂s), mean(gmaxs)])

                end

                png(p, "$(plot_path)/DONE_xs_iter$(iter)noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")

            else

                synth_δ = 5
                ks = DiscreteNonParametric(collect(xmin:xmax), [1 / (xmax - xmin + 1) for _ in xmin:xmax])

                while maximum(ks.p) < maximum(10 / (xmax - xmin), 0.12)

                    synth_n = rand(ks)
                    pos = 0
                    for k in 1:K

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
                        maximum([(pos + 1) / (K + 2), 1 - (pos + 1) / (K + 2)])
                    )
                    @show p̂
                    l_mult = p̂ ^ (pos + 1) * (1 - p̂) ^ (K + 2 - pos)
                    @show l_mult
                    r_mult = (1 - p̂) ^ (pos + 1) * p̂ ^ (K + 2 - pos)
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

                png(p, "$(plot_path)/DONE_xs_iter$(iter)noise$(λ)realn$(real_n)model$(model_name)metric$(metric)")

            end

        end

        open("$(out_path)/$(myid())_tps.csv", "a") do io
            write(io, "$(iter),$(λ),$(real_n),$(model_name),$(metric),$(xmin),$(xmid₁),$(xmid₂),$(xmax)\n")
        end

    end

end

run_experiment()