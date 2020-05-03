using SpecialFunctions
using StatsFuns: log1pexp, log2π
using MLJBase: auc

"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
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
    ps = mean(map(θ -> logistic.(X_test * θ), samples[1:25:end]))
    if plot_roc
        plot_roc_curve(y_test, ps)
    end
    return roc_auc(y_test, ps), log_loss(X_test, y_test, samples), avg_pdf(X_test, y_test, samples)
end


function roc_auc(ys, ps)
    auc([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
end


function log_loss(X, y, samples)
    return -mean(map(θ -> sum(logpdf_bernoulli_logit.(X_test * θ, y_test)), samples[1:25:end]))
end


function avg_pdf(X, y, samples)
    return mean(map(θ -> sum(pdf_bernoulli_logit.(X_test * θ, y_test)), samples[1:25:end]))
end


function create_bayes_factor_matrix(bayes_factors)
    matrix = DataFrame([bf ./ bayes_factors for bf in bayes_factors])
    rename!(matrix, [:mlj, :beta, :weighted, :naive, :no_synth])
    return matrix
end


function result_storage()
    return DataFrame(
        real_α = Float64[],
        synth_α = Float64[],
        mlj_auc = Float64[],
        beta_auc = Float64[],
        weighted_auc = Float64[],
        naive_auc = Float64[],
        no_synth_auc = Float64[],
        mlj_ll = Float64[],
        beta_ll = Float64[],
        weighted_ll = Float64[],
        naive_ll = Float64[],
        no_synth_ll = Float64[],
    )
end
