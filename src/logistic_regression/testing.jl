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
include("src/logistic_regression/utils.jl")
include("src/logistic_regression/experiment.jl")
include("src/logistic_regression/mathematical_utils.jl")
include("src/logistic_regression/distributions.jl")
include("src/logistic_regression/weight_calibration.jl")
include("src/logistic_regression/evaluation.jl")

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
temp = Matrix(CSV.read("real.csv"))
y = temp[:, 15]
X = temp[:, 1:14]

lr2 = LogisticRegression(λ, 0.; fit_intercept = false)
θ_0 = MLJLinearModels.fit(lr2, X, (2 .* y) .- 1; solver=MLJLinearModels.LBFGS())
weight_calib(X,y,β,θ_0)


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
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> beta_loss(X[i, :]', y[i], β, θ), big.(θ̂))
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    w_data = sum(diag(Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data))) / sum(diag(Jθ̂_data))

    return w_data
end
