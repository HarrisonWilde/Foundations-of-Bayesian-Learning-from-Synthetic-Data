using Turing, Distributions
using RDatasets
using MCMCChains, Plots, StatsPlots
using CSV
using Random
using LinearAlgebra

# Set a seed for reproducibility.
Random.seed!(0)

# Import the "Default" dataset.
data = RDatasets.dataset("ISLR", "Default")

# Convert "Default" and "Student" to numeric values.
data[!,:DefaultNum] = [r.Default == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]
data[!, :StudentNum] = [r.Student == "Yes" ? 1.0 : 0.0 for r in eachrow(data)]

# Delete the old columns which say "Yes" and "No".
select!(data, Not([:Default, :Student]))

# Function to split samples.
function split_data(df, at = 0.70)
    (r, _) = size(df)
    index = Int(round(r * at))
    train = df[1:index, :]
    test  = df[(index+1):end, :]
    return train, test
end

# Rescale our columns.
data.Balance = (data.Balance .- mean(data.Balance)) ./ std(data.Balance)
data.Income = (data.Income .- mean(data.Income)) ./ std(data.Income)

CSV.write("data/raw/islr.csv", data)

real_train = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_real_train.csv")
real_test = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_real_test.csv")
synth_train = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_synth_train.csv")
synth_test = CSV.read("data/splits/islr_DefaultNum_split0.05_ganpate_eps6.0_synth_test.csv")



# Split our dataset 5/95 into training/test sets.
# train, test = split_data(data, 0.05);

# Create our labels. These are the values we are trying to predict.
y_real_train = real_train[:,:DefaultNum]
y_real_test = real_test[:,:DefaultNum]

# Remove the columns that are not our predictors.
X_real_train = Matrix(real_train[:,[:StudentNum, :Balance, :Income]])
X_real_test = Matrix(real_test[:,[:StudentNum, :Balance, :Income]])

# Create our labels. These are the values we are trying to predict.
y_synth_train = synth_train[:,:DefaultNum]
y_synth_test = synth_test[:,:DefaultNum]

# Remove the columns that are not our predictors.
X_synth_train = Matrix(synth_train[:,[:StudentNum, :Balance, :Income]])
X_synth_test = Matrix(synth_test[:,[:StudentNum, :Balance, :Income]])

# Bayesian logistic regression (LR)
@model logistic_regression(X_real, y_real, X_synth, y_synth, m, n_real, n_synth, σ) = begin
    Θ ~ MvNormal(m + 1, σ)
    for i in 1:n_real
        v_real = logistic(Θ[1] + dot(X_real[i, :], Θ[2:end]))
        y_real[i] ~ Bernoulli(v_real)
    end
    for i in 1:n_synth
        v_synth = logistic(Θ[1] + dot(X_synth[i, :], Θ[2:end]))
        y_synth[i] ~ Bernoulli(v_synth)
    end
end

# Retrieve the number of observations.
n_real, m = size(X_real_train)
n_synth, _ = size(X_synth_train)
σ = 100.0

samples, stats = sample(
    logistic_regression(X_real_train, y_real_train, X_synth_train, y_synth_train, m, n_real, n_synth, σ),
    HMC(0.05, 10),
    1500
)

# Sample using HMC.
chain = mapreduce(
    c -> sample(
        logistic_regression(X_real_train, y_real_train, X_synth, y_synth, m, σ),
        HMC(0.05, 10),
        1500
    ),
    chainscat,
    1:3
)

describe(chain)

plot(chain)

# The labels to use.
l = [:student, :balance, :income]

# Use the corner function. Requires StatsPlots and MCMCChains.
corner(chain, l)

function prediction(x::Matrix, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain[:intercept].value)
    student = mean(chain[:student].value)
    balance = mean(chain[:balance].value)
    income = mean(chain[:income].value)

    # Retrieve the number of rows.
    n, _ = size(x)

    # Generate a vector to store our predictions.
    v = Vector{Float64}(undef, n)

    # Calculate the logistic function for each element in the test set.
    for i in 1:n
        num = logistic(intercept .+ student * x[i,1] + balance * x[i,2] + income * x[i,3])
        if num >= threshold
            v[i] = 1
        else
            v[i] = 0
        end
    end
    return v
end

# Set the prediction threshold.
threshold = 0.10

# Make the predictions.
predictions = prediction(test, chain, threshold)

# Calculate MSE for our test set.
loss = sum((predictions - test_label).^2) / length(test_label)

defaults = sum(test_label)
not_defaults = length(test_label) - defaults

predicted_defaults = sum(test_label .== predictions .== 1)
predicted_not_defaults = sum(test_label .== predictions .== 0)

println("Defaults: $$defaults
    Predictions: $$predicted_defaults
    Percentage defaults correct $$(predicted_defaults/defaults)")

println("Not defaults: $$not_defaults
    Predictions: $$predicted_not_defaults
    Percentage non-defaults correct $$(predicted_not_defaults/not_defaults)")
