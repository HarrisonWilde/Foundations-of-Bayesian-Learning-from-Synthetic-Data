# Need to define the loss on an uncontrained paramater space
function weight_calib(X, y, β, λ)

    f(θ) = βloss(X, y, β, θ)
    lr = LogisticRegression(λ; fit_intercept = false)
    θ_0 = MLJLinearModels.fit(lr, X, y; solver = MLJLinearModels.LBFGS())
    @show θ_0
    @show f(θ_0)
    @show Hβloss(y' * X, β, θ_0)
    n, p = size(X)
    # θ̂ = θ_0
    θ̂ = Optim.minimizer(optimize(f, θ_0, Optim.LBFGS(); autodiff=:forward))
    @show θ̂
    @show f(θ̂)
    grad_data = zeros(n, p)
    Hess_data = zeros(p, p, n)
    mean_grad_sq_data = zeros(p, p)
    mean_Hess_data = zeros(p, p)

    for i in 1:n
        grad_data[i, :] = ∇βloss(X[i, :]' * y[i], β, θ̂)
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = Hβloss(X[i, :]' * y[i], β, θ̂)
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    w_data = sum(diag(Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data))) / sum(diag(Jθ̂_data))

    return w_data
end
