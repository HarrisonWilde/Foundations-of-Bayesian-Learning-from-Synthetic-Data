function weight_calib(experiment_type, βw_default)
    
    try
        if experiment_type == "gaussian"
            βw = gaussian_weight_calib()
        elseif experiment_type == "logistic_regression"
            βw = logistic_weight_calib()
        end
  
        if isnan(βw)
            βw = βw_default
        elseif βw > 5
            βw = 5
        elseif βw < 0.5
            βw = 0.5
        end
    catch
        βw = βw_default
        print("Calibration failed")
    end
    return [βw]

end


# Need to define the loss on an uncontrained paramater space
function gaussian_weight_calib(y, β, αₚ, βₚ, μₚ, σₚ)

    σ₀ = √rand(InverseGamma(αₚ, βₚ))
    μ₀ = rand(Distributions.Normal(μₚ, σₚ * σ₀))
    n, p = length(y), 2
    f(θ) = gaussian_βloss(θ..., β, y)
    μ̂, σ̂ = Optim.minimizer(optimize(f, [μ₀, σ₀]; autodiff=:forward))

    grad_data = zeros(n, p)
    Hess_data = zeros(p, p, n)
    mean_grad_sq_data = zeros(p, p)
    mean_Hess_data = zeros(p, p)

    for i in 1:n
        grad_data[i, :] = ForwardDiff.gradient(θ -> gaussian_βloss(θ..., β, y[i]), [μ̂, σ̂])
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> gaussian_βloss(θ..., β, y[i]), [μ̂, σ̂])
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    w_data = sum(diag(Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data))) / sum(diag(Jθ̂_data))

    return w_data
end


# Need to define the loss on an uncontrained paramater space
function logistic_weight_calib(X, y, β, λ)

    f(θ) = logistic_βloss(X, y, β, θ)
    lr = LogisticRegression(λ; fit_intercept = false)
    θ_0 = MLJLinearModels.fit(lr, X, y; solver = MLJLinearModels.LBFGS())
    @show θ_0
    @show f(θ_0)
    @show logistic_Hβloss(y' * X, β, θ_0)
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
        grad_data[i, :] = logistic_∇βloss(X[i, :]' * y[i], β, θ̂)
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = logistic_Hβloss(X[i, :]' * y[i], β, θ̂)
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    w_data = sum(diag(Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data))) / sum(diag(Jθ̂_data))

    return w_data
end
