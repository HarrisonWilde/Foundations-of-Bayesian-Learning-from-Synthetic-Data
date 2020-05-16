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
include("../common/utils.jl")
include("../common/weight_calibration.jl")
include("distributions.jl")
include("loss.jl")
include("evaluation.jl")
include("init.jl")

name, label, ε, folds, split, distributed, use_ad = "uci_heart", "target", "6.0", 5, 1.0, true, false
t = Dates.format(now(), "HH_MM_SS__dd_mm_yyyy")
println("Setting up experiment...")
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

n_samples, n_warmup = 10000, 2000
show_progress = true
labels, real_data, synth_data = load_data(name, label, ε)
θ_dim = size(real_data)[2] - 1
c = classes(categorical(real_data[:, labels[1]])[1])

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
y_r = (2 .* y_real) .- 1
y_s = (2 .* y_synth) .- 1

chn1 = sample(β_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw), Turing.NUTS(), 10000)
chn2 = sample(weighted_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, w), Turing.NUTS(), 10000)
chn3 = sample(naive_model(X_real, X_synth, y_real, y_synth, θ_dim, σ), Turing.NUTS(), 10000)
chn4 = sample(no_synth_model(X_real, y_real, θ_dim, σ), Turing.NUTS(), 10000)
describe(chn, sections=:internals)

yX_real = y_r .* X_real
yX_synth = y_s .* X_synth
yXθ_real = yX_real * initial_θ
Xθ_synth = X_synth * initial_θ
yXθ_synth = y_s .* Xθ_synth
ℓπ_β(θ) = (
    ℓpdf_MvNorm(σ, θ) +
    sum(ℓpdf_BL.(yX_real * θ)) +
    βw * sum(ℓpdf_βBL.(X_synth * θ, y_s, β))
)
∇ℓπ_β(θ) = (
    ℓπ_β(θ),
    ∇ℓpdf_MvNorm(σ, θ) +
    ∇ℓpdf_BL(yX_real, θ) +
    βw * ∇ℓpdf_βBL(yX_synth, β, θ)
)

ℓπ_w(θ) = (
    ℓpdf_MvNorm(σ, θ) +
    sum(ℓpdf_BL.(yX_real * θ)) +
    w * sum(ℓpdf_BL.(yX_synth * θ))
)
∇ℓπ_w(θ) = (
    ℓπ_w(θ),
    ∇ℓpdf_MvNorm(σ, θ) +
    ∇ℓpdf_BL(yX_real, θ) +
    w * ∇ℓpdf_BL(yX_synth, θ)
)

ℓπ(θ) = (
    ℓpdf_MvNorm(σ, θ) +
    sum(ℓpdf_BL.(yX_real * θ)) +
    sum(ℓpdf_BL.(yX_synth * θ))
)
∇ℓπ(θ) = (
    ℓπ(θ),
    ∇ℓpdf_MvNorm(σ, θ) +
    ∇ℓpdf_BL(yX_real, θ) +
    ∇ℓpdf_BL(yX_synth, θ)
)

ℓπ_ns(θ) = (
    ℓpdf_MvNorm(σ, θ) +
    sum(ℓpdf_BL.(yX_real * θ))
)
∇ℓπ_ns(θ) = (
    ℓπ_ns(θ),
    ∇ℓpdf_MvNorm(σ, θ) +
    ∇ℓpdf_BL(yX_real, θ)
)












temp = Matrix(CSV.read("real.csv"))
y = temp[:, 15]
X = temp[:, 1:14]
y_stat = y
lr2 = LogisticRegression(λ, 0.; fit_intercept = false)
y_ml = (2 .* y) .- 1
y = y_ml
θ_0 = MLJLinearModels.fit(lr2, X, y_ml; solver=MLJLinearModels.LBFGS())
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
