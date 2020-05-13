
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
    logistic_z = logistic.(z)
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
    # θ̂ = θ_0
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
    w_data = sum(diag((Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data)))) / sum(diag(Jθ̂_data))

    return w_data
end
