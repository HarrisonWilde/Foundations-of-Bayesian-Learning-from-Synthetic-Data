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
using Random: seed!
using StatsFuns: log1pexp, log2π
using MLJBase: auc
include("utils.jl")
include("experiment.jl")
include("mathematical_utils.jl")
include("distributions.jl")
include("weight_calibration.jl")
include("evaluation.jl")

name, label, ε, folds, split, distributed, use_ad = "uci_heart", "target", "6.0", 5, 1.0, true, false
t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
println("Loading data...")
labels, real_data, synth_data = load_data(name, label, ε)
println("Setting up experiment...")
θ_dim = size(real_data)[2] - 1
w = 0.5
β = 0.5
βw = 1.15
σ = 50.0
λ = 0.0
real_αs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 0.75]
synth_αs = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75]
αs = get_conditional_pairs(real_αs, synth_αs)
num_αs = size(αs)[1]
total_steps = num_αs * folds
# results = SharedArray{Float64, 2}("results", (total_steps, 10))
# bayes_factors = SharedArray{Float64, 3}("bayes_factors", (4, 4, total_steps))

n_samples, n_warmup = 15000, 5000
show_progress = true

i = 628

fold = ((i - 1) % folds)
real_α, synth_α = αs[Int(ceil(i / folds))]
X_real, y_real, X_synth, y_synth, X_valid, y_valid = fold_α(
    real_data, synth_data, real_α, synth_α,
    fold, folds, labels
)
metric, initial_θ = init_run(
    θ_dim, λ, X_real, y_real, X_synth, y_synth, β
)
lr2 = LogisticRegression(λ, 0.; fit_intercept = false)
θ_0 = MLJLinearModels.fit(lr2, X_synth, y_synth; solver=MLJLinearModels.LBFGS())
X, y = X_synth, y_synth



function beta_loss(X, y, β, θ)

    z = X * θ
    logistic_z = logistic.(z)
    loss = -sum(@. (1.0 / β) * (
        y * logistic(z) + (1 - y) * (1 - logistic(z))
    ) ^ β - (1.0 / (β + 1.0)) * (
        logistic_z ^ (β + 1.0)
        + (1.0 - logistic_z) ^ (β + 1.0)
    ))
    return loss

end


function ∂beta_loss∂θ(X, y, β, θ)

    z = X * θ
    pdf = pdf_bernoulli_logit.(z, Int(y))
    ∂logistic_zX = @. ∂logistic(z) * X
    ∂ℓpdf∂θ = @. y * logistic(-z) * X - (1.0 - y) * logistic(z) * X
    ∂loss∂θ = -vec(βw * sum((@. pdf ^ β * (
            ∂ℓpdf∂θ
        ) - (
            logistic_z ^ β * ∂logistic_zX - (1.0 - logistic_z) ^ β * ∂logistic_zX
        )),
        dims=1
    ))
    return ∂loss∂θ

end


# Need to define the loss on an uncontrained paramater space
function weight_calib(X, y, β, θ_0)
    # θ_0 = [4.41720489, 0.41632245, -23.38931654, 11.71856821, -0.42375295, -0.06698908, -4.55647240, 3.97981790, 0.57330941, 3.02025206, -10.33997373, 7.97994416, -10.13166633, -13.37909493]
    θ_0 = [5.325730848, 0.025682393, -3.542510715, 1.460540432, -0.048783089, -0.004425257, 0.041810460, 0.899856071, 0.042562678, -0.185233851, -1.252361784, 1.029490076, -1.064326000, -1.583962267 ]
    n, p = size(X)
    f(θ_0) = beta_loss(X, y, β, θ_0)
    # g!(θ) = ∂beta_loss∂θ(∂loss∂θ, X, y, β, θ)
    θ̂ = Optim.minimizer(optimize(f, θ_0, Optim.LBFGS(); autodiff=:forward))

    grad_data = zeros(n, p)
    Hess_data = zeros(p, p, n)
    mean_grad_sq_data = zeros(p, p)
    mean_Hess_data = zeros(p, p)

    for i in 1:n
        grad_data[i, :] = ForwardDiff.gradient(θ -> beta_loss(X[i, :]', y[i], β, θ), θ̂)
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> beta_loss(X[i, :]', y[i], β, θ), θ̂)
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    @show Iθ̂_data
    @show Jθ̂_data
    w_data = sum(diag((Jθ̂_data .* inv(Iθ̂_data) .* transpose(Jθ̂_data)))) / sum(diag(Jθ̂_data))

    return w_data
end
