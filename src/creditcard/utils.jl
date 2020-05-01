using SpecialFunctions
using StatsFuns: log1pexp, log2π

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


function evaluate()
    print("Nothing")
end
