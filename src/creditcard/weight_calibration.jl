
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
    # θ_0 = [4.41720489, 0.41632245, -23.38931654, 11.71856821, -0.42375295, -0.06698908, -4.55647240, 3.97981790, 0.57330941, 3.02025206, -10.33997373, 7.97994416, -10.13166633, -13.37909493]
    θ_0 = [5.325730848, 0.025682393, -3.542510715, 1.460540432, -0.048783089, -0.004425257, 0.041810460, 0.899856071, 0.042562678, -0.185233851, -1.252361784, 1.029490076, -1.064326000, -1.583962267 ]
    n, p = size(X)
    f(θ_0) = beta_loss(X, y, β, θ_0)
    # g!(θ) = ∂beta_loss∂θ(∂loss∂θ, X, y, β, θ)
    θ̂ = Optim.minimizer(optimize(f, θ_0, Optim.LBFGS(); autodiff=:forward))

    grad_data = zeros(n, p)
    Hess_data = zeros(p, p, n)
    qwertyuiop = zeros(p, p)
    mean_Hess_data = zeros(p, p)

    for i in 1:n
        grad_data[i, :] = ForwardDiff.gradient(θ -> beta_loss(X[i, :]', y[i], β, θ), θ̂)
        qwertyuiop += (grad_data[i, :] .* transpose(grad_data[i, :]))
        Hess_data[:, :, i] = ForwardDiff.hessian(θ -> beta_loss(X[i, :]', y[i], β, θ), θ̂)
        mean_Hess_data += Hess_data[:, :, i]
    end

    Iθ̂_data = qwertyuiop ./ n
    Jθ̂_data = mean_Hess_data ./ n
    @show Iθ̂_data
    @show Jθ̂_data
    w_data = sum(diag((Jθ̂_data .* inv(Iθ̂_data) .* transpose(Jθ̂_data)))) / sum(diag(Jθ̂_data))

    return w_data
end
