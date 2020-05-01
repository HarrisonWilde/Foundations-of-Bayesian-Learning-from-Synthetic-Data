using SpecialFunctions
using StatsFuns: log1pexp, log2π
using MLJBase: auc

"""
Returns pairs of elements from two separate lists, provided their sum is < max_sum (default 1 for proportions)
"""
function get_conditional_pairs(l1, l2; max_sum=1)
    return ((a1, a2) for a1 in l1 for a2 in l2 if a1 + a2 <= max_sum)
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
    # ŷ0 = exp.(log.(sum(map(θ -> exp.(logpdf_bernoulli_logit.(X_test * θ, y_test)), samples_β))) .- log(size(samples_β)[1]))
    # ŷ = mean(map(θ -> pdf_bernoulli_logit.(X_test * θ, y_test), samples))
    ps = mean(map(θ -> logistic.(X_test * θ), samples))
    if plot_roc
        plot_roc_curve(y_test, ps)
    end
    return roc_auc(y_test, ps)
end


function roc_auc(ys, ps)
    auc([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
end


function result_storage()
    return DataFrame(
        real_α = Float64[],
        synth_α = Float64[],
        mlj = Float64[],
        β = Float64[],
        weighted = Float64[],
        naive = Float64[],
        no_synth = Float64[],
    )
end
