using Turing, ForwardDiff, LinearAlgebra, StatsFuns

function ℓπ_beta(σ, β, βw, θ, X_real, y_real, X_synth, y_synth)
    D = size(θ)[1]
    ℓprior = sum(logpdf.(MvNormal(D, σ), θ))
    ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    z_synth = X_synth * θ
    ℓsynth = βw * sum(
        (1 / β) * pdf.(BinomialLogit.(1, z_synth), y_synth) .^ β
        - (1 / (β + 1)) .* (
            pdf.(BinomialLogit.(1, z_synth), y_synth) .^ (β + 1)
            .+ (1 .- pdf.(BinomialLogit.(1, z_synth), y_synth)) .^ (β + 1)
        )
    )
    return (ℓprior + ℓreal + ℓsynth)
end


function ℓπ_kld(σ, w, θ, X_real, y_real, X_synth, y_synth)
    D = size(θ)[1]
    ℓprior = sum(logpdf(MvNormal(D, σ), θ))
    ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
    ℓsynth = w * sum(logpdf.(BinomialLogit.(1, X_synth * θ), y_synth))
    return (ℓprior + ℓreal + ℓsynth)
end


function ∂logistic(z::Float64)
    return exp(-z) / (1 + exp(-z)) ^ 2
end


function ∂pdf∂θ(X::Array{Float64,2}, y::Array{Int64,1}, z::Array{Float64,1}, θ::Array{Float64,1})
    ∂log = ∂logistic.(z)
    return transpose(X) * (y .* ∂log) - transpose(X) * ((1.0 .- y) .* ∂log)
end


function ∂ℓπ∂θ_beta(σ, β, βw, θ, X_real, y_real, X_synth, y_synth)
    D = size(θ)[1]
    Σ = abs2(σ) * I + zeros(D, D)
    ∂ℓprior∂θ = sum(-inv(Σ) * θ)
    z_real = X_real * θ
    # ℓreal = sum(log.(y_real .* (1 ./ (1 .+ exp.(-X_real * θ))) + (1 .- y_real) .* (1 ./ (1 .+ exp.(X_real * θ)))))
    ∂ℓreal∂θ = sum(transpose(y_real .* StatsFuns.logistic.(-z_real)) * X_real - transpose((1.0 .- y_real) .* StatsFuns.logistic.(z_real)) * X_real)
    # ∂pdf∂θ = transpose(y_real .* ∂logistic.(X_real * θ)) * X_real - transpose((1.0 .- y_real) .* ∂logistic.(X_real * θ)) * X_real
    z_synth = X_synth * θ
    # TODO
    deriv = ∂pdf∂θ(X_synth, y_synth, z_synth, θ)
    _pdf = pdf.(BinomialLogit.(1, z_synth), y_synth)
    ∂ℓsynth∂θ = βw * sum(
        _pdf .^ β .* deriv
        - (1) .* (
            _pdf .^ (β)
            .- (1 .- _pdf) .^ (β)
        ) .* deriv
    )
    # ∂ℓsynth∂θ = βw * sum(
    #     ∂pdf∂θ(X_synth, y_synth, z_synth, θ) .^ (β - 1)
    #     -1 .* (
    #         ∂pdf∂θ(X_synth, y_synth, z_synth, θ) .^ β
    #         .+ (1 - ∂pdf∂θ(X_synth, y_synth, z_synth, θ)) .^ β
    #     )
    # )
end


function ∂ℓπ∂θ_kld(σ, w, θ, X_real, y_real, X_synth, y_synth)
    D = size(θ)[1]
    Σ = abs2(σ) * I + zeros(D, D)
    ∂ℓprior∂θ = sum(-inv(Σ) * θ)
    z_real = X_real * θ
    ∂ℓreal∂θ = sum(transpose(y_real .* StatsFuns.logistic.(-z_real)) * X_real - transpose((1.0 .- y_real) .* StatsFuns.logistic.(z_real)) * X_real)
    z_synth = X_synth * θ
    ∂ℓsynth∂θ = w * sum(transpose(y_synth .* StatsFuns.logistic.(-z_synth)) * X_synth - transpose((1.0 .- y_synth) .* StatsFuns.logistic.(z_synth)) * X_synth)
    return (∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
end

σ = 100
θ = [0.1, 0.3, 0.4, 0.2]
w = 0.5
β = 0.5
βw = 1.1
X_real = round.(5 * (rand(5, 4) .- 0.5), digits=1)
y_real = [1, 0, 1, 0, 1]
X_synth = X_real + round.(0.5 * (rand(5, 4) .- 0.5), digits=1)
y_synth = [1, 0, 1, 0, 1]
D = size(θ)[1]
Σ = abs2(σ) * I + zeros(D, D)
z_real = X_real * θ
z_synth = X_synth * θ

# To test gradients are right vvvv

# ℓprior = sum(logpdf(MvNormal(D, σ), θ))
ForwardDiff.gradient(θ -> sum(logpdf(MvNormal(D, σ), θ)), θ)
-inv(Σ) * θ

# ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
ForwardDiff.gradient(θ -> sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real)), θ)
# logpdf.(BinomialLogit(X_real, y_real, θ)) = sum(log.(y_real .* (1 ./ (1 .+ exp.(-X_real * θ))) + (1 .- y_real) .* (1 ./ (1 .+ exp.(X_real * θ)))))
transpose(y_real .* StatsFuns.logistic.(-z_real)) * X_real -transpose((1.0 .- y_real) .* StatsFuns.logistic.(z_real)) * X_real


Distributions.pdf(d::BinomialLogit{<:Real}, k::Int) = exp(logpdf(d, k))
ForwardDiff.gradient(θ -> sum(pdf.(BinomialLogit.(1, X_real * θ), y_real)), θ)
ForwardDiff.gradient(θ -> sum(y_real .* (1 ./ (1 .+ exp.(-X_real * θ))) + (1 .- y_real) .* (1 .- (1 ./ (1 .+ exp.(-X_real * θ))))), θ)
ForwardDiff.gradient(θ -> sum(y_real .* StatsFuns.logistic.(X_real * θ) + (1 .- y_real) .* (1 .- StatsFuns.logistic.(X_real * θ))), θ)
transpose(X_real) * (y_real .* ∂logistic.(X_real * θ)) - transpose(X_real) * ((1.0 .- y_real) .* ∂logistic.(X_real * θ))


deriv = ∂pdf∂θ(X_synth, y_synth, z_synth, θ)
_pdf = pdf.(BinomialLogit.(1, z_synth), y_synth)
ℓsynth1(θ) = βw * sum(
    (1 / β) * pdf.(BinomialLogit.(1, X_synth * θ), y_synth) .^ β
    - (1 / (β + 1)) .* (
        pdf.(BinomialLogit.(1, X_synth * θ), y_synth) .^ (β + 1)
        .+ (1 .- pdf.(BinomialLogit.(1, X_synth * θ), y_synth)) .^ (β + 1)
    )
)
ℓsynth2(θ) = βw * sum(
    (1 / β) * (
        y_real .* StatsFuns.logistic.(X_real * θ) + (1 .- y_real) .* (1 .- StatsFuns.logistic.(X_real * θ))) .^ β
    - (1 / (β + 1)) .* (
        StatsFuns.logistic.(X_real * θ) .^ (β + 1)
        .+ (1 .- StatsFuns.logistic.(X_real * θ)) .^ (β + 1)
    )
)
ForwardDiff.gradient(θ -> ℓsynth1(θ), θ)
ForwardDiff.gradient(θ -> ℓsynth2(θ), θ)

for k in size(X_real)[2]

∂ℓsynth∂θ = βw * sum(
    _pdf .^ β .* deriv
    - (1) .* (
        _pdf .^ (β)
        .- (1 .- _pdf) .^ (β)
    ) .* deriv
)


# p_logistic = exp(0.5*y[i,1]*lin_pred[i,1]) / (exp(0.5*lin_pred[i,1]) + exp(-0.5*lin_pred[i,1]))

# target += 1/beta_p * p_logistic^beta_p - 1/(beta_p+1) * (p_logistic^(beta_p + 1) + (1 - p_logistic)^(beta_p + 1))

#  SEBASTIAN STUFF BELOW

# === pdf stuff ===

# ∂logistic(s::Float64)=(exp(-s))/(1+exp(-s))^2
# function π(θ::Vector)
#   atsFuns.logistic.(X_real * θ)
#   ℓreal= sum((probs .* y_real + (1.0 .- probs) .* (1.0.-y_real ))
# end

# g(θ::Vector)= ForwardDiff.gradient(v -> ℓπ( v), θ)

# function pdf_grad(θ::Vector) # gradient pdf
#   transpose(y_real .* Dlogistic.(X_real * θ)) X_real -transpose((1.0 .- y_real) *. Dlogistic.(X_real * θ)) *X_real
# end



# === log pdf stuff ===
#((log f)'= f'/f)
# function ℓπ(θ::Vector)
#   probs = StatsFuns.logistic.(X_real * θ)
#   ℓreal = sum(log.(probs .* y_real + (1.0 .- probs) .* (1.0 .- y_real)))
# end

# d = log(1-logistic) =  - d logistic /( exp(-s)/ (1+exp(-s))
#  =   exp(-s)/ (1+exp(-s))^2 / ( exp(-s)/ (1+exp(-s)= - logistic

# z = X_real * θ
# dℓrealdθ = transpose(y_real .* StatsFuns.logistic.(-z)) * X_real -transpose((1.0 .- y_real) .* StatsFuns.logistic.(z)) * X_real

# function logpdf_grad(θ::Vector)
#   transpose(y_real .* dloglogistic.(X_real * θ)) X_real -transpose((1.0 .- y_real) . logistic.(X_real * θ)) *X_real
# end