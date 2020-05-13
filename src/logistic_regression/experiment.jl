function fold_α(real_data, synth_data, real_α, synth_α, fold, folds, labels)
    len_real = size(real_data)[1]
    len_synth = size(synth_data)[1]
    real_chunk = len_real / folds
    synth_chunk = len_synth / folds

    if fold == 0
        real_ix = collect(floor(Int, 1 + real_chunk):floor(Int, real_chunk * (1 + (folds - 1) * real_α)))
        synth_ix = collect(floor(Int, 1 + synth_chunk):floor(Int, real_chunk * (1 + (folds - 1) * synth_α)))
    elseif fold == folds - 1
        real_ix = collect(1:floor(Int, (fold) * real_chunk * real_α))
        synth_ix = collect(1:floor(Int, (fold) * synth_chunk * synth_α))
    else
        real_ix = vcat(
            collect(floor(Int, 1 + ((fold + 1) * real_chunk)):len_real),
            collect(1:floor(Int, fold * real_chunk))
        )[1:floor(Int, real_chunk * (folds - 1) * real_α)]
        synth_ix = vcat(
            collect(floor(Int, 1 + ((fold + 1) * synth_chunk)):len_synth),
            collect(1:floor(Int, fold * synth_chunk))
        )[1:floor(Int, synth_chunk * (folds - 1) * synth_α)]
    end

    X_real = Matrix(real_data[real_ix, Not(labels)])
    y_real = Int.(real_data[real_ix, labels[1]])
    X_synth = Matrix(synth_data[synth_ix, Not(labels)])
    y_synth = Int.(synth_data[synth_ix, labels[1]])
    X_valid = Matrix(real_data[
        floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
        Not(labels)
    ])
    y_valid = Int.(real_data[
        floor(Int, 1 + (fold * real_chunk)):floor(Int, (fold + 1) * real_chunk),
        labels[1]
    ])
    return X_real, y_real, X_synth, y_synth, X_valid, y_valid
end


function init_stan_models(n_samples, n_warmup)

    β_model = SampleModel(
        "Beta",
        open(f -> read(f, String), "src/logistic_regression/stan/BetaLogisticRegression.stan");
        method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup)
    )
    weighted_model = SampleModel(
        "Weighted",
        open(f -> read(f, String), "src/logistic_regression/stan/WeightedStandardLogisticRegression.stan");
        method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup)
    )
    naive_model = SampleModel(
        "Naive",
        open(f -> read(f, String), "src/logistic_regression/stan/StandardLogisticRegression.stan");
        method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup)
    )
    no_synth_model = SampleModel(
        "NoSynth",
        open(f -> read(f, String), "src/logistic_regression/stan/NoSynthLogisticRegression.stan");
        method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup)
    )
    return β_model, weighted_model, naive_model, no_synth_model

end


"""
Define the mass matrix, make an initial guess at θ at the MLE using MLJ's LogiticRegression and calibrate βw
"""
function init_run(θ_dim, λ, X_real, y_real, X_synth, y_synth, β; use_zero_init=false)

    # Define mass matrix, initial guess at θ
    metric = DiagEuclideanMetric(θ_dim)
    if use_zero_init
        initial_θ = zeros(θ_dim)
    else
        lr1 = LogisticRegression(λ, 0.; fit_intercept = false)
        initial_θ = MLJLinearModels.fit(lr1, X_real, (2 .* y_real) .- 1; solver=MLJLinearModels.LBFGS())
        # auc_mlj, ll_mlj, bf_mlj = evalu(X_test, y_test, [initial_θ])
        lr2 = LogisticRegression(λ; fit_intercept = false)
        θ_0 = MLJLinearModels.fit(lr2, X_synth, (2 .* y_synth) .- 1; solver=MLJLinearModels.LBFGS())
    end
    return metric, initial_θ
end


function setup_run(ℓπ, ∂ℓπ∂θ, metric, initial_θ; use_ad=true, target_acceptance_rate=0.75)

    # Setup a Hamiltonian system
    hamiltonian = Hamiltonian(
        metric,
        ℓπ,
        use_ad ? ForwardDiff : ∂ℓπ∂θ,
    )

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ; max_n_iters=10000)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with multinomial sampling scheme, generalised No-U-Turn criteria, and windowed adaption for step-size and diagonal mass matrix
    proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    # proposal = AdvancedHMC.StaticTrajectory(integrator, 100)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(target_acceptance_rate, integrator))

    return hamiltonian, proposal, adaptor
end



#
# print("Starting...")
#
# # real_α = 0.1
# # synth_α = 0.1
# @sync begin
#     # this task prints the progress bar
#     @async while take!(channel)
#         next!(p)
#     end
#
#     @async begin
#         # @showprogress 1 for i in 1:num_αs
#         @distributed for i in 1:num_αs
#
#             # real_α, synth_α = αs[i]
#             # # Take matrix slices according to αs
#             # X_real = Matrix(real_train[1:floor(Int32, len_real * real_α), Not(labels)])
#             # y_real = Int.(real_train[1:floor(Int32, len_real * real_α), labels[1]])
#             # X_synth = Matrix(synth_train[1:floor(Int32, len_synth * synth_α), Not(labels)])
#             # y_synth = Int.(synth_train[1:floor(Int32, len_synth * synth_α), labels[1]])
#
#
#
#         end
#         put!(channel, false)
#     end
# end
