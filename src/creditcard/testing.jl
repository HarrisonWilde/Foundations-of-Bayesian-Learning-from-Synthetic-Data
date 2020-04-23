using Turing
using ForwardDiff
using LinearAlgebra
using StatsFuns
include("src/creditcard/utils.jl")
include("src/creditcard/densities.jl")

σ = 100
θ = [0.1, 0.3, 0.4, 0.2]
w = 0.5
β = 0.5
βw = 1.0
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

∂k = zeros(size(X_synth)[2])
for k in 1:size(X_synth)[2]
    for i in 1:size(X_synth)[1]
        zi = dot(X_synth[i, :], θ)
        ∂kℓi = (
            (StatsFuns.logistic(zi) * y_synth[i] + (1 - StatsFuns.logistic(zi)) * (1 - y_synth[i])) ^ (β - 1)
            * (∂logistic(zi) * X_synth[i, k] * y_synth[i] - ∂logistic(zi) * X_synth[i, k] * (1 - y_synth[i]))
            - (StatsFuns.logistic(zi) ^ β * ∂logistic(zi) * X_synth[i, k] - (1 - StatsFuns.logistic(zi)) ^ β * ∂logistic(zi) * X_synth[i, k])
        )
        ∂k[k] += ∂kℓi
    end
end
print(∂k)

ForwardDiff.gradient(θ -> sum(
        (1 / β) .* pdf.(BinomialLogit.(1, X_synth * θ), y_synth) .^ β
        - (1 / (β + 1)) .* (
            StatsFuns.logistic.(X_synth * θ) .^ (β + 1)
            .+ (1 .- StatsFuns.logistic.(X_synth * θ)) .^ (β + 1)
        )
    ), θ)

z_synth = X_synth * θ
sum((
    pdf.(BinomialLogit.(1, z_synth), y_synth) .^ (β - 1)
    .* (∂logistic.(z_synth) .* X_synth .* y_synth - ∂logistic.(z_synth) .* X_synth .* (1 .- y_synth))
    .- (StatsFuns.logistic.(z_synth) .^ β .* ∂logistic.(z_synth) .* X_synth .- (1 .- StatsFuns.logistic.(z_synth)) .^ β .* ∂logistic.(z_synth) .* X_synth)
), dims=1)



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