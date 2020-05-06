function beta_loss(X, y, β, θ)
    if length(size(X)) == 1
        z = dot(X, θ)
    else
        z = X * θ
    end
    logistic_z = logistic.(z)
    loss = -sum(@. (1.0 / β) * (
        y * logistic(z) + (1 - y) * (1 - logistic(z))
    ) ^ β - (1.0 / (β + 1.0)) * (
        logistic_z ^ (β + 1.0)
        + (1.0 - logistic_z) ^ (β + 1.0)
    ))
    return loss
end


# Need to define the loss on an uncontrained paramater space
function weight_calib(X, y, β, θ_0)
    n, p = size(X)
    #theta_hat<-optim(initial_θ,function(theta){loss(data,theta)},gr=function(theta){grad(function(theta){loss(data,theta)},theta)},method="BFGS")
    θ̂ = Optim.minimizer(optimize(θ -> beta_loss(X, y, β, θ), θ_0, BFGS(); autodiff=:forward))

    grad_data = Array{Float64, 2}(undef, (n, p))
    Hess_data = Array{Float64, 3}(undef, (p, p, n))
    mean_grad_sq_data = zeros(p, p)
    mean_Hess_data = zeros(p, p)
    for i in 1:n
        grad_data[i, :] = ForwardDiff.gradient(θ -> beta_loss(X[i, :], y[i], β, θ), θ̂)
        mean_grad_sq_data += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> beta_loss(X[i, :], y[i], β, θ), θ̂)
        mean_Hess_data += Hess_data[:, :, i]
    end
    Iθ̂_data = mean_grad_sq_data ./ n
    Jθ̂_data = mean_Hess_data ./ n
    # Figure out how to express this in julia, could do inv(Iθ̂_data) but...
    w_data = sum(diag((Jθ̂_data .* inv(Iθ̂_data) .* transpose(Jθ̂_data)))) / sum(diag(Jθ̂_data))
    return w_data
end
