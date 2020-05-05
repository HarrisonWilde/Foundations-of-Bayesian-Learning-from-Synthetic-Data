
"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return vcat([(a1, a2) for a1 ∈ l1 for a2 ∈ l2 if a1 + a2 <= max_sum], [(a, max_sum - a) for a ∈ l1 if max_sum - a ∉ l2])
end

"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_valid_synth_αs(real_α, synth_αs; max_sum=1)
    if max_sum - real_α in synth_αs
        return [synth_α for synth_α in synth_αs if synth_α + real_α <= max_sum]
    else
        return vcat([synth_α for synth_α in synth_αs if synth_α + real_α <= max_sum], [max_sum - real_α])
    end
end


#  SHOULD THIS BE -SUM OR JUST SUM? I THINK I GET WHY IT IS MINUS BUT CHECK
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


function setup_run(ℓπ, ∂ℓπ∂θ, metric, initial_θ; manual=true, target_acceptance_rate=0.75)

    # Setup a Hamiltonian system
    hamiltonian = Hamiltonian(
        metric,
        ℓπ,
        manual ? ∂ℓπ∂θ : Zygote,
    )

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with multinomial sampling scheme, generalised No-U-Turn criteria, and windowed adaption for step-size and diagonal mass matrix
    proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_acceptance_rate, integrator))

    return hamiltonian, proposal, adaptor
end


function evalu(X_test, y_test, samples; plot_roc=false)
    ps = probabilities(X_test, samples)
    if plot_roc
        plot_roc_curve(y_test, ps)
    end
    return roc_auc(y_test, ps), log_loss(X_test, y_test, samples), marginal_likelihood_estimate(X_test, y_test, samples)
end


function roc_auc(ys, ps)
    auc([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
end


function probabilities(X, samples)
    N = size(samples)[1]
    avg = logistic.(X * samples[1]) ./ N
    for θ in samples[2:end]
        avg += logistic.(X * θ) ./ N
    end
    return avg
end


function log_loss(X, y, samples)
    N = size(samples)[1]
    avg = 0
    for θ in samples
        avg += sum(logpdf_bernoulli_logit.(X * θ, y)) / N
    end
    return -avg
end


# https://www.jstor.org/stable/pdf/2291091.pdf?refreqid=excelsior%3Ab194b370e4efc9f1d9ae29b7c7c5c6da
function marginal_likelihood_estimate(X, y, samples)
    N = size(samples)[1]
    avg = 0
    for θ in samples
        avg += sum(pdf_bernoulli_logit.(X * θ, y) ^ -1) / N
    end
    return avg ^ -1
    # mean(map(θ -> sum(pdf_bernoulli_logit.(X_test * θ, y_test)), samples))
end


function create_bayes_factor_matrix(bayes_factors)
    return [bf1 / bf2 for bf1 ∈ bayes_factors, bf2 ∈ bayes_factors]
end


function create_results_df(results)
    df = DataFrame(results)
    rename!(df, [:real_α, :synth_α, :mlj_auc, :beta_auc, :weighted_auc, :naive_auc, :no_synth_auc, :mlj_ll, :beta_ll, :weighted_ll, :naive_ll, :no_synth_ll])
    return df
end
