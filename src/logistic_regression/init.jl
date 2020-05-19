function init_stan_models(model_names, n_samples, n_warmup; dist = true)

    models = [(
        "$(name)_$(myid())",
        SampleModel(
            "$(name)_$(myid())",
            open(f -> read(f, String), "src/logistic_regression/stan/$(name)_logistic_regression.stan");
            method = StanSample.Sample(num_samples=n_samples - n_warmup, num_warmup=n_warmup),
            tmpdir = dist ? "$(@__DIR__)/tmp/" : mktempdir()
        )
    ) for name in model_names]
    return models

end


struct log_posterior_gradient_pair{T, U}
    ℓπ::T
    ∇ℓπ::U
end


function init_ahmc_models(X_real, y_real, X_synth, y_synth, σ, w, βw, β, initial_θ)

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
    return [
        ("beta", log_posterior_gradient_pair(ℓπ_β, ∇ℓπ_β)),
        ("weighted", log_posterior_gradient_pair(ℓπ_w, ∇ℓπ_w)),
        ("naive", log_posterior_gradient_pair(ℓπ, ∇ℓπ)),
        ("no_synth", log_posterior_gradient_pair(ℓπ_ns, ∇ℓπ_ns))
    ]

end


function init_turing_models(X_real, y_real, X_synth, y_synth, σ, w, βw, β)

    return [
        ("beta", β_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw)),
        ("weighted", weighted_model(X_real, X_synth, y_real, y_synth, θ_dim, σ, w)),
        ("naive", naive_model(X_real, X_synth, y_real, y_synth, θ_dim, σ)),
        ("no_synth", no_synth_model(X_real, y_real, θ_dim, σ))
    ]

end


"""
Define the mass matrix, make an initial guess at θ at the MLE using MLJ's LogiticRegression and calibrate βw
"""
function init_run(λ, X_real, y_real, X_synth, y_synth, β; use_zero_init=false)

    # initial guess at θ
    if use_zero_init
        initial_θ = zeros(θ_dim)
    else
        lr1 = LogisticRegression(λ; fit_intercept = false)
        initial_θ = MLJLinearModels.fit(lr1, X_real, y_real; solver = MLJLinearModels.LBFGS())
        lr2 = LogisticRegression(λ; fit_intercept = false)
        θ_0 = MLJLinearModels.fit(lr2, X_synth, y_synth; solver = MLJLinearModels.LBFGS())
    end
    return initial_θ
end
