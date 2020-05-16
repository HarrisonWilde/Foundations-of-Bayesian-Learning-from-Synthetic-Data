# function beta_loss(X, y, β, θ)
#
#     z = X * θ
#     logistic_z = logistic.(z)
#     loss = -sum(@. (1.0 / β) * (
#         y * logistic(z) + (1 - y) * (1 - logistic(z))
#     ) ^ β - (1.0 / (β + 1.0)) * (
#         logistic_z ^ (β + 1.0)
#         + (1.0 - logistic_z) ^ (β + 1.0)
#     ))
#     return loss
#
# end
#
#
# function ∂beta_loss∂θ(X, y, β, θ)
#
#     z = X * θ
#     pdf = pdf_bernoulli_logit.(z, Int.(y))
#     logistic_z = logistic.(z)
#     ∂logistic_zX = @. ∂logistic(z) * X
#     ∂ℓpdf∂θ = @. y * logistic(-z) * X - (1.0 - y) * logistic(z) * X
#     ∂loss∂θ = -vec(βw * sum((@. pdf ^ β * (
#             ∂ℓpdf∂θ
#         ) - (
#             logistic_z ^ β * ∂logistic_zX - (1.0 - logistic_z) ^ β * ∂logistic_zX
#         )),
#         dims=1
#     ))
#     return ∂loss∂θ
#
# end


function βloss(X, y, β, θ)

    yXθ = y .* X * θ
    p = logistic.(yXθ)
    loss = @. 1 / β * p ^ β - 1 / (β + 1) * (p ^ (β + 1) + logistic(-yXθ) ^ (β + 1))
    return -sum(loss)

end


function yXβ(yX::T, β::U, θ::V) where {T, U, V}
    eyXθ = exp.(-yX * θ)
    return yX' * (eyXθ ./ (1 .+ eyXθ) .^ β)
end
function ∇βloss(yX, β, θ)
    return -yXβ(yX, β + 1, θ) + yXβ(yX, β + 2, θ) + yXβ(-yX, β + 2, θ)
end

function βyXᵀyX(X, y, β, θ)

    eyXθ = exp.(-y .* X * θ)
    return (y .* X)' * (y .* X) .* (
        (β .* eyXθ .^ 2) ./ ((1 .+ eyXθ) .^ (β + 1))
        - (eyXθ ./ (1 .+ eyXθ) .^ β)
    )

end


function Hβloss(X, y, β, θ)

    hessian_lik_term = βyXᵀyX(X, y, β + 1, θ)
    hessian_int_term = βyXᵀyX(X, y, β + 2, θ) + βyXᵀyX(X, -y, β + 2, θ)
    return -hessian_lik_term + hessian_int_term

end