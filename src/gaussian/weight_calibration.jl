# Need to define the loss on an uncontrained paramater space
function weight_calib(y, β, μ_0, σ²_0)

    n, p = length(y), 2
    f(θ) = βloss(θ..., β, y)
    μ̂, σ̂² = Optim.minimizer(optimize(f, [μ_0, σ²_0]; autodiff=:forward))

    grad_data = zeros(n, p)
    Hess_data = zeros(p, p, n)
    mean_grad_sq_data = zeros(p, p)
    mean_Hess_data = zeros(p, p)

    for i in 1:n
        grad_data[i, :] = ForwardDiff.gradient(θ -> βloss(θ..., β, y[i]), [μ̂, σ̂²])
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> βloss(θ..., β, y[i]), [μ̂, σ̂²])
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    w_data = sum(diag(Jθ̂_data * inv(Iθ̂_data) * transpose(Jθ̂_data))) / sum(diag(Jθ̂_data))

    return w_data
end
