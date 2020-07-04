invϕ = (√5 - 1) / 2
invϕ² = (3 - √5) / 2

function parse_cl()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--path", "-p"
            help = "specify the path to the top of the project"
            arg_type = String
            required = true
        "--dataset", "-d"
            help = "specify the dataset to be used"
            arg_type = String
        "--label", "-l"
            help = "specify the label to be used, must be present in dataset, obviously"
            arg_type = String
        "--epsilon"
            help = "choose an epsilon value: the differential privacy constant"
            arg_type = String
            default = "6.0"
        "--iterations", "-i"
            help = "number of full iterations to run"
            arg_type = Int
            default = 1
        "--folds", "-k"
            help = "specify the number of CV folds to be carried out during training"
            arg_type = Int
            default = 5
        "--split", "-s"
            help = "specify the ratio of the data to be used for training, the rest will be held back for testing"
            arg_type = Float64
            default = 1.0
        "--use_ad"
            help = "include to use ForwardDiff rather than manually defined derivatives"
            action = :store_true
        "--distributed", "-c"
            help = "include when running the code in a distributed fashion"
            action = :store_true
        "--sampler", "-o"
            help = "choose from AHMC, Turing and CmdStan"
            arg_type = String
            default = "Turing"
        "--no_shuffle"
            help = "Disable row shuffling on load of data"
            action = :store_true
        "--scales"
            help = "List of scales to use for Laplace noise"
            arg_type = Float64
            nargs = '*'
        "--num_repeats"
            arg_type = Int
            default = 100
        "--algorithm", "-a"
            arg_type = String
            default = "golden"
        "--experiment", "-e"
            arg_type = String
            default = "logistic_regression"
    end
    return parse_args(s)
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
