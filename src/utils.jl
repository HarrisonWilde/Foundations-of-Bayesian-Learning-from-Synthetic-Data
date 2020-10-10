invϕ = (√5 - 1) / 2
invϕ² = (3 - √5) / 2


"""
Return the logistic function computed in a numerically stable way:
``logistic(x) = 1/(1+exp(-x))``
"""
function LOGISTIC(T)
    log(one(T) / Base.eps(T) - one(T))
end
function logistic(x::T) where {T}
    LOGISTIC_T = LOGISTIC(T)
    x > LOGISTIC_T && return one(x)
    x < -LOGISTIC_T && return zero(x)
    return one(x) / (one(x) + exp(-x))
end

"""
Derivative of the logistic function
"""
function ∂logistic(z::T) where {T}
	a = exp(z) + 1
    return (1 / a) - (1 / (a ^ 2))
end


function dotmany(X, Θ, groups, nₚ)
    res = similar(groups, typeof(zero(eltype(X)) * zero(eltype(Θ))))
    @inbounds for obs in eachindex(groups)
        Θi = groups[obs]
        y = zero(eltype(res))
        for p in 1:nₚ
            y += X[obs, p] * Θ[p, Θi]
        end
        res[obs] = y
    end
    return res
end


function to_float(l)
    return map(x -> parse(Float64, x), l)
end


function parse_cl()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--experiment_type"
            arg_type = String
            default = "gaussian"
            required = true
        "--id"
            arg_type = String
            required = true
        "--path"
            help = "specify the path to the top of the project"
            arg_type = String
            default = "."
        "--seed"
            arg_type = Int
            default = Int(round(rand() * 1e12))
        "--show_progress"
            action = :store_true
        "--iterations"
            help = "number of full iterations to run"
            arg_type = Int
            default = 1
        "--split"
            help = "specify the ratio of the data to be used for training, the rest will be held back for testing"
            arg_type = Float64
            default = 1.0
        "--use_ad"
            help = "include to use ForwardDiff rather than manually defined derivatives"
            action = :store_true
        "--distributed"
            help = "include when running the code in a distributed fashion"
            action = :store_true
        "--n_samples"
            help = "number of MCMC samples to take"
            arg_type = Int
            default = 10000
        "--n_warmup"
            arg_type = Int
            default = 1000
        "--n_chains"
            arg_type = Int
            default = 1
        "--target_acceptance"
            arg_type = Float64
            default = 0.8
        "--sampler"
            help = "choose from AHMC, Turing and CmdStan"
            arg_type = String
            default = "Turing"
        "--betas", "-b"
            nargs = '*'
            arg_type = Float64
            default = [0.5]
        "--beta_weights"
            nargs = '*'
            arg_type = Float64
            default = [1.25]
        "--calibrate_beta_weight"
            action = :store_true
        "--weights", "-w"
            nargs = '*'
            arg_type = Float64
            default = [0.0, 0.5, 1.0]
        "--metrics"
            nargs = '*'
            arg_type = String
        "--model_names"
            nargs = '*'
            arg_type = String
        # (Logistic) Regression specific
        "--dataset"
            help = "specify the dataset to be used"
            arg_type = String
        "--label"
            help = "specify the label to be used, must be present in dataset, obviously"
            arg_type = String
        "--epsilon"
            help = "choose an epsilon value: the differential privacy constant"
            arg_type = String
            default = "6.0"
        "--real_alphas"
            nargs = '*'
            arg_type = Float64
        "--real_alpha_range"
            nargs = '*'
            arg_type = Float64
        "--synth_alphas"
            nargs = '*'
            arg_type = Float64
        "--synth_alpha_range"
            nargs = '*'
            arg_type = Float64
        "--folds"
            help = "specify the number of CV folds to be carried out during training"
            arg_type = Int
            default = 5
        # Gaussian specific
        "--mu"
            arg_type = Float64
            default = 0.0
        "--sigma"
            arg_type = Float64
            default = 1.0
        "--scales"
            help = "List of scales to use for Laplace noise"
            arg_type = Float64
            nargs = '*'
        "--real_ns"
            nargs = '*'
            arg_type = Int
        "--synth_ns"
            nargs = '*'
            arg_type = Int
        "--real_n_range"
            nargs = '*'
            arg_type = Int
        "--synth_n_range"
            nargs = '*'
            arg_type = Int
        "--n_unseen"
            arg_type = Int
            default = 500
        "--algorithm"
            arg_type = String
            default = "golden"
        "--alphas"
            arg_type = Float64
            nargs = '*'
        "--fn"
            arg_type = Int
            default = 0
        ### Golden Section specific
        "--num_repeats"
            arg_type = Int
            default = 100
        # Priors
        "--mu_p"
            arg_type = Float64
            default = 0.0
        "--sigma_p"
            arg_type = Float64
            default = 1.0
        "--alpha_p"
            arg_type = Float64
            default = 0.1
        "--beta_p"
            arg_type = Float64
            default = 0.1
        "--nu_p"
            arg_type = Float64
            default = 3.0
        "--Sigma_p"
            arg_type = Matrix
        
    end
    return parse_args(s)
end


function config_dict(experiment_type, args)

    if experiment_type in ["logistic_regression", "regression"]
        config = (
            real_alphas = length(args["real_alpha_range"]) == 3 ? vcat(collect(args["real_alpha_range"][1]:args["real_alpha_range"][2]:args["real_alpha_range"][3]), args["real_alphas"]) : args["real_alphas"],
            synth_alphas = length(args["synth_alpha_range"]) == 3 ? vcat(collect(args["synth_alpha_range"][1]:args["synth_alpha_range"][2]:args["synth_alpha_range"][3]), args["synth_alphas"]) : args["synth_alphas"],
            folds = args["folds"],
            metrics = args["metrics"]
        )
    elseif experiment_type == "gaussian"
        config = (
            real_ns = length(args["real_n_range"]) == 3 ? vcat(collect(args["real_n_range"][1]:args["real_n_range"][2]:args["real_n_range"][3]), args["real_ns"]) : args["real_ns"],
            synth_ns = length(args["synth_n_range"]) == 3 ? vcat(collect(args["synth_n_range"][1]:args["synth_n_range"][2]:args["synth_n_range"][3]), args["synth_ns"]) : args["synth_ns"],
            n_unseen = args["n_unseen"],
            λs = args["scales"],
            algorithm = args["algorithm"],
            metrics = args["metrics"]
        )
    end
    return config

end


function generate_model_configs(model_names, βs, βws, ws, αs)

    if length(αs) > 0
        weighted_model_pairs = [(
            model = m, weight = w, β = -1
        ) for m ∈ model_names for w ∈ αs if m == "weighted"]
    else
        weighted_model_pairs = [(
            model = m, weight = w, β = -1
        ) for m ∈ model_names for w ∈ ws if m == "weighted"]
    end
    beta_model_pairs = [(
        model = m, weight = w, β = b
    ) for m ∈ model_names for w ∈ βws for b ∈ βs if m ∈ ["beta", "beta_all"]]
    resampled_model_pairs = [(
        model = m, weight = 1., β = -1
    ) for m ∈ model_names if m == "resampled"]
    other_model_pairs = [(
        model = m, weight = -1, β = -1,
    ) for m ∈ model_names if m ∉ ["beta", "beta_all", "weighted", "resampled"]]
    model_pairs = vcat(weighted_model_pairs, beta_model_pairs, resampled_model_pairs, other_model_pairs)

end


function load_data(name, label, ε)

    labels = [Symbol(label),]
    real_data = CSV.read("data/splits/$(name)_$(label)_eps$(ε)_real.csv")
    synth_data = CSV.read("data/splits/$(name)_$(label)_eps$(ε)_synth.csv")

    # Append 1s column to each to allow for intercept term in logistic regression
    real_data = hcat(DataFrame(intercept = ones(size(real_data)[1])), real_data)
    synth_data = hcat(DataFrame(intercept = ones(size(synth_data)[1])), synth_data)

    return labels, real_data, synth_data

end


function standardise_out(data)

    for name in names(data)

        if all(isequal(first(data[name])), data[name]) || (minimum(data[name]) >= 0 && maximum(data[name]) <= 1)
            continue
        end
        data[name] = float(data[name])
        data[name] .-= mean(data[name])
        data[name] ./= std(data[name])

    end
    data

end


function generate_all_steps(experiment_type, algorithm, iterations, config, model_configs)

    if experiment_type in ["logistic_regression", "regression"]
        # S = Iterators.product(
        #     iterations,
        #     config[:real_alphas],
        #     config[:synth_alphas],
        #     config[:folds],
        #     model_configs
        # )
        S = [
            (a, b, c, d, e)
            for a ∈ iterations
            for b ∈ config[:real_alphas]
            for c ∈ config[:synth_alphas]
            for d ∈ [i for i ∈ 0:(config[:folds]-1)]
            for e ∈ model_configs
        ]
    else
        if (experiment_type == "gaussian") & (algorithm != "basic")
            S = [
                (a, b, c, d, e, f, g)
                for a ∈ iterations
                for b ∈ config[:real_ns]
                for c ∈ config[:synth_ns]
                for d ∈ config[:n_unseen]
                for e ∈ config[:λs]
                for f ∈ model_configs
                for g ∈ config[:metrics]
            ]
        else
            S = [
                (a, b, c, d, e, f)
                for a ∈ iterations
                for b ∈ config[:real_ns]
                for c ∈ config[:synth_ns]
                for d ∈ config[:n_unseen]
                for e ∈ config[:λs]
                for f ∈ model_configs
            ]
        end
    end
    return S

end


function fold_α(real_data, synth_data, real_α, synth_α, fold, folds, labels; predictors = [], groups = [], add_intercept = false, continuous_y = false)

    len_real = size(real_data)[1]
    len_synth = size(synth_data)[1]
    real_chunk = len_real / folds
    synth_chunk = len_synth / folds

    if fold == 0
        real_ix = collect(floor(Int, 1 + real_chunk):floor(Int, real_chunk * (1 + (folds - 1) * real_α)))
        synth_ix = collect(floor(Int, 1 + synth_chunk):floor(Int, real_chunk * (1 + (folds - 1) * synth_α)))
    elseif fold == folds - 1
        real_ix = collect(1:floor(Int, (fold) * real_chunk * real_α))
        synth_ix = collect(1:floor(Int, (fold) * synth_chunk * synth_α))
    else
        real_ix = vcat(
            collect(floor(Int, 1 + ((fold + 1) * real_chunk)):len_real),
            collect(1:floor(Int, fold * real_chunk))
        )[1:floor(Int, real_chunk * (folds - 1) * real_α)]
        synth_ix = vcat(
            collect(floor(Int, 1 + ((fold + 1) * synth_chunk)):len_synth),
            collect(1:floor(Int, fold * synth_chunk))
        )[1:floor(Int, synth_chunk * (folds - 1) * synth_α)]
    end

    if length(predictors) == 0

        X_real = Matrix(real_data[real_ix, Not(labels)])
        X_synth = Matrix(synth_data[synth_ix, Not(labels)])
        X_valid = Matrix(real_data[
            floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
            Not(labels)
        ])

    else

        X_real = Matrix(real_data[real_ix, predictors])
        X_synth = Matrix(synth_data[synth_ix, predictors])
        X_valid = Matrix(real_data[
            floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
            predictors
        ])

    end

    if add_intercept

        X_real = hcat(ones(size(X_real)[1]), X_real)
        X_synth = hcat(ones(size(X_synth)[1]), X_synth)
        X_valid = hcat(ones(size(X_valid)[1]), X_valid)

    end

    if continuous_y

        y_real = Array(real_data[real_ix, labels[1]])
        y_synth = Array(synth_data[synth_ix, labels[1]])
        y_valid = Array(real_data[
            floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
            labels[1]
        ])

    else

        y_real = Int.(real_data[real_ix, labels[1]])
        y_synth = Int.(synth_data[synth_ix, labels[1]])
        y_valid = Int.(real_data[
            floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
            labels[1]
        ])

    end

    if length(groups) == 0
        return X_real, y_real, X_synth, y_synth, X_valid, y_valid
    else
        groups_real = Array(real_data[real_ix, groups[1]])
        groups_synth = Array(synth_data[synth_ix, groups[1]])
        groups_valid = Array(real_data[
            floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
            groups[1]
        ])
        return X_real, y_real, groups_real, X_synth, y_synth, groups_synth, X_valid, y_valid, groups_valid
    end
end


function setup_run(ℓπ, metric, initial_θ; ∂ℓπ∂θ=ForwardDiff, target_acceptance_rate=0.8)

    # Setup a Hamiltonian system
    hamiltonian = Hamiltonian(
        metric,
        ℓπ,
        ∂ℓπ∂θ,
    )

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ; max_n_iters=1000)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with multinomial sampling scheme, generalised No-U-Turn criteria, and windowed adaption for step-size and diagonal mass matrix
    proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    # proposal = AdvancedHMC.StaticTrajectory(integrator, 100)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_acceptance_rate, integrator))

    return hamiltonian, proposal, adaptor

end


"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return vcat([(a1, a2) for a1 ∈ l1 for a2 ∈ l2 if a1 + a2 <= max_sum], [(a, max_sum - a) for a ∈ l1 if max_sum - a ∉ l2])
end


"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_valid_synth_αs(real_α, synth_αs; max_sum=1)
    if max_sum - real_α in synth_αs
        return [synth_α for synth_α in synth_αs if synth_α + real_α <= max_sum]
    else
        return vcat([synth_α for synth_α in synth_αs if synth_α + real_α <= max_sum], [max_sum - real_α])
    end
end


function create_bayes_factor_matrix(bayes_factors)
    return [bf1 / bf2 for bf1 ∈ bayes_factors, bf2 ∈ bayes_factors]
end


function create_results_df(results)
    df = DataFrame(results)
    rename!(df, [:real_α, :synth_α, :mlj_auc, :beta_auc, :weighted_auc, :naive_auc, :no_synth_auc, :mlj_ll, :beta_ll, :weighted_ll, :naive_ll, :no_synth_ll])
    return df
end
