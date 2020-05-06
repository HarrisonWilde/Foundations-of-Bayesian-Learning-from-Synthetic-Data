function parse_cl()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
            help = "specify the dataset to be used"
            arg_type = String
            required = true
        "--label", "-l"
            help = "specify the label to be used, must be present in dataset, obviously"
            arg_type = String
            required = true
        "--epsilon", "-e"
            help = "choose an epsilon value: the differential privacy constant"
            arg_type = String
            default = "6.0"
        "--folds", "-k"
            help = "specify the number of CV folds to be carried out during training"
            arg_type = Int
            required = true
        "--split", "-s"
            help = "specify the ratio of the data to be used for training, the rest will be held back for testing"
            arg_type = Float64
            default = 1.0
        "--use_ad", "-ad"
            help = "include to use Zygote rather than manually defined derivatives"
            action = :store_true
        "--distributed", "-c"
            help = "include when running the code in a distributed fashion"
            action = :store_true
    end
    return parse_args(s)
end


function load_data(name, label, eps; shuffle_rows=true)

    labels = [Symbol(label),]
    real_data = CSV.read("data/splits/$(name)_$(label)_eps$(eps)_real.csv")
    synth_data = CSV.read("data/splits/$(name)_$(label)_eps$(eps)_synth.csv")

    # Append 1s column to each to allow for intercept term in logistic regression
    real_data = hcat(DataFrame(intercept = ones(size(real_data)[1])), real_data)
    synth_data = hcat(DataFrame(intercept = ones(size(synth_data)[1])), synth_data)

    if shuffle_rows
        real_data = real_data[shuffle(axes(real_data, 1)), :]
        synth_data = synth_data[shuffle(axes(synth_data, 1)), :]
    end

    return labels, real_data, synth_data
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
