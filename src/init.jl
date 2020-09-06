function init_stan_models(path, experiment_type, sampler, n_samples, n_warmup, n_chains, model_names, target_acceptance_rate; dist = true)

    tmpdir = dist ? "$(path)/tmp_$(experiment_type)_$(sampler)/" : mktempdir()
    if sampler == "Stan"
        models = [(
            "$(name)_$(myid())",
            SampleModel(
                "$(name)_$(myid())",
                open(
                    f -> read(f, String),
                    "src/stan_models/$(name)_$(experiment_type).stan"
                ),
                n_chains = n_chains,
                tmpdir = tmpdir,
                method = StanSample.Sample(
                    num_samples = n_samples - n_warmup,
                    num_warmup = n_warmup,
                    adapt = StanSample.Adapt(delta=target_acceptance_rate)
                )
            )
        ) for name in model_names]
    elseif sampler == "CmdStan"
        models = [(
            "$(name)_$(myid())",
            Stanmodel(
                CmdStan.Sample(
                    num_samples = n_samples - n_warmup,
                    num_warmup = n_warmup,
                    adapt = CmdStan.Adapt(delta=target_acceptance_rate)
                );
                name = "$(name)_$(myid())",
                nchains = n_chains,
                model = open(
                    f -> read(f, String),
                    "src/stan_models/$(name)_$(experiment_type).stan"
                ),
                tmpdir = tmpdir,
                output_format = :mcmcchains
            )
        ) for name in model_names]
    end
    return OrderedDict(models)

end


# function init_stan_models(model_names, n_samples, n_warmup; dist = true)

#     models = [(
#         "$(name)_$(myid())",
#         SampleModel(
#             "$(name)_$(myid())",
#             open(f -> read(f, String), "src/logistic_regression/stan/$(name)_logistic_regression.stan");
#             method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup),
#             tmpdir = dist ? "$(@__DIR__)/tmp/" : mktempdir()
#         )
#     ) for name in model_names]
#     return models

# end


struct log_posterior_gradient_pair{T, U}
    ℓπ::T
    ∇ℓπ::U
end


function init_ahmc_gaussian_models(real_data, synth_data, w, βw, β, λ, αₚ, βₚ, μₚ, σₚ)

    b = inv(Bijectors.stack(Identity{0}(), bijector(InverseGamma(αₚ, βₚ))))

    ℓπ_β(θ) = (
        logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
        ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
        sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
        βw * sum(ℓpdf_βN(θ[1], √θ[2], β, synth_data))
    )
    ℓπ_β_unconstrained(θ_unconstrained) = (ℓπ_β ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)

    ℓπ_w(θ) = (
        logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
        ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
        sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
        w * sum(ℓpdf_N(θ[1], √θ[2], synth_data))
    )
    ℓπ_w_unconstrained(θ_unconstrained) = (ℓπ_w ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)

    ℓπ_βa(θ) = (
        logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
        ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
        βw * sum(ℓpdf_βN(θ[1], √θ[2], β, real_data)) +
        βw * sum(ℓpdf_βN(θ[1], √θ[2], β, synth_data))
    )
    ℓπ_βa_unconstrained(θ_unconstrained) = (ℓπ_βa ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)

    ℓπ_na(θ) = (
        logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
        ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
        sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
        sum(ℓpdf_NL.(θ[1], √θ[2], λ, synth_data))
    )
    ℓπ_na_unconstrained(θ_unconstrained) = (ℓπ_na ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)

    return b, OrderedDict([
        ("beta", ℓπ_β_unconstrained),
        ("weighted", ℓπ_w_unconstrained),
        ("beta_all", ℓπ_βa_unconstrained),
        ("noise_aware", ℓπ_na_unconstrained)
    ])

end

function init_ahmc_logistic_models(X_real, y_real, X_synth, y_synth, σ, w, βw, β, initial_θ)

    yX_real = y_real .* X_real
    yX_synth = y_synth .* X_synth
    Xθ_synth = X_synth * initial_θ
    yXθ_real = yX_real * initial_θ
    yXθ_synth = y_synth .* Xθ_synth
    ℓπ_β(θ) = (
        ℓpdf_MvNorm(σ, θ) +
        sum(ℓpdf_BL.(yX_real * θ)) +
        βw * sum(ℓpdf_βBL.(X_synth * θ, y_synth, β))
    )
    ∇ℓπ_β(θ) = (
        ℓπ_β(θ),
        ∇ℓpdf_MvNorm(σ, θ) +
        ∇ℓpdf_BL(yX_real, θ) +
        βw * ∇ℓpdf_βBL(yX_synth, β, θ)
    )

    ℓπ_w(θ) = (
        ℓpdf_MvNorm(σ, θ) +
        sum(ℓpdf_BL.(yX_real * θ)) +
        w * sum(ℓpdf_BL.(yX_synth * θ))
    )
    ∇ℓπ_w(θ) = (
        ℓπ_w(θ),
        ∇ℓpdf_MvNorm(σ, θ) +
        ∇ℓpdf_BL(yX_real, θ) +
        w * ∇ℓpdf_BL(yX_synth, θ)
    )

    ℓπ(θ) = (
        ℓpdf_MvNorm(σ, θ) +
        sum(ℓpdf_BL.(yX_real * θ)) +
        sum(ℓpdf_BL.(yX_synth * θ))
    )
    ∇ℓπ(θ) = (
        ℓπ(θ),
        ∇ℓpdf_MvNorm(σ, θ) +
        ∇ℓpdf_BL(yX_real, θ) +
        ∇ℓpdf_BL(yX_synth, θ)
    )

    ℓπ_ns(θ) = (
        ℓpdf_MvNorm(σ, θ) +
        sum(ℓpdf_BL.(yX_real * θ))
    )
    ∇ℓπ_ns(θ) = (
        ℓπ_ns(θ),
        ∇ℓpdf_MvNorm(σ, θ) +
        ∇ℓpdf_BL(yX_real, θ)
    )
    return OrderedDict([
        ("beta", log_posterior_gradient_pair(ℓπ_β, ∇ℓπ_β)),
        ("weighted", log_posterior_gradient_pair(ℓπ_w, ∇ℓπ_w)),
        ("naive", log_posterior_gradient_pair(ℓπ, ∇ℓπ)),
        ("no_synth", log_posterior_gradient_pair(ℓπ_ns, ∇ℓπ_ns))
    ])

end


function init_turing_gaussian_models(real_data, synth_data, w, βw, β, λ, αₚ, βₚ, μₚ, σₚ)

    return OrderedDict([
        ("beta", β_gaussian_model(real_data, synth_data, βw, β, αₚ, βₚ, μₚ, σₚ)),
        ("weighted", weighted_gaussian_model(real_data, synth_data, w, αₚ, βₚ, μₚ, σₚ)),
        # ("naive", naive_gaussian_model(real_data, synth_data, αₚ, βₚ, μₚ, σₚ)),
        # ("no_synth", no_synth_gaussian_model(real_data, αₚ, βₚ, μₚ, σₚ)),
        ("beta_all", β_all_gaussian_model(real_data, synth_data, βw, β, αₚ, βₚ, μₚ, σₚ)),
        ("noise_aware", noise_aware_gaussian_model(real_data, synth_data, λ, αₚ, βₚ, μₚ, σₚ))
    ])

end

function init_turing_logistic_models(X_real, y_real, X_synth, y_synth, σ, w, βw, β)

    return OrderedDict([
        ("beta", β_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw)),
        ("weighted", weighted_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, w)),
        ("naive", naive_logistic_model(X_real, X_synth, y_real, y_synth, θ_dim, σ)),
        ("no_synth", no_synth_logistic_model(X_real, y_real, θ_dim, σ))
    ])

end

function init_turing_regression_models(
    y_real, X_real, groups_real,
    y_synth, X_synth, groups_synth,
    αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
    nₚ, nₛ, β, βw, w)

    return OrderedDict([
        (
            "beta",
            β_regression_model(
                y_real, X_real, groups_real,
            	y_synth, X_synth, groups_synth,
            	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
            	nₚ, nₛ, β, βw
            )
        ),
        (
            "weighted",
            weighted_regression_model(
                y_real, X_real, groups_real,
                y_synth, X_synth, groups_synth,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ, w
            )
        ),
        (
            "naive",
            naive_regression_model(
                y_real, X_real, groups_real,
                y_synth, X_synth, groups_synth,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ
            )
        ),
        (
            "no_synth",
            no_synth_regression_model(
                y_real, X_real, groups_real,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ
            )
        ),
    ])

end


"""
Define the mass matrix, make an initial guess at θ at the MLE using MLJ's LogiticRegression and calibrate βw
"""
function init_run(λ, X_real, y_real, β; use_zero_init=false)

    # initial guess at θ
    if use_zero_init
        initial_θ = zeros(θ_dim)
    else
        lr = LogisticRegression(λ; fit_intercept = false)
        initial_θ = MLJLinearModels.fit(lr, X_real, y_real; solver = MLJLinearModels.LBFGS())
    end
    return initial_θ
end

"""
Generate data according to the synthetic DGP for Normal-Laplace
"""
function gen_synth(n, dgp, λ)
    return rand(dgp, n) + rand(Laplace(0, λ), n)
end
