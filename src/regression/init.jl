# function init_ahmc_models(real_data, synth_data, w, βw, β, λ, αₚ, βₚ, μₚ, σₚ)
#
#     b = inv(Bijectors.stack(Identity{0}(), bijector(InverseGamma(αₚ, βₚ))))
#
#     ℓπ_β(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
#         βw * sum(ℓpdf_βN(θ[1], √θ[2], β, synth_data))
#     )
#     ℓπ_β_unconstrained(θ_unconstrained) = (ℓπ_β ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     ℓπ_w(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
#         w * sum(ℓpdf_N(θ[1], √θ[2], synth_data))
#     )
#     ℓπ_w_unconstrained(θ_unconstrained) = (ℓπ_w ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     ℓπ(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
#         sum(ℓpdf_N(θ[1], √θ[2], synth_data))
#     )
#     ℓπ_unconstrained(θ_unconstrained) = (ℓπ ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     ℓπ_ns(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         sum(ℓpdf_N(θ[1], √θ[2], real_data))
#     )
#     ℓπ_ns_unconstrained(θ_unconstrained) = (ℓπ_ns ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     ℓπ_βa(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         βw * sum(ℓpdf_βN(θ[1], √θ[2], β, real_data)) +
#         βw * sum(ℓpdf_βN(θ[1], √θ[2], β, synth_data))
#     )
#     ℓπ_βa_unconstrained(θ_unconstrained) = (ℓπ_βa ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     ℓπ_na(θ) = (
#         logpdf(InverseGamma(αₚ, βₚ), θ[2]) +
#         ℓpdf_N(μₚ, σₚ * √θ[2], θ[1]) +
#         sum(ℓpdf_N(θ[1], √θ[2], real_data)) +
#         sum(ℓpdf_NL.(θ[1], √θ[2], λ, synth_data))
#     )
#     ℓπ_na_unconstrained(θ_unconstrained) = (ℓπ_na ∘ b)(θ_unconstrained) + logabsdetjac(b, θ_unconstrained)
#
#     return b, OrderedDict([
#         ("beta", ℓπ_β_unconstrained),
#         ("weighted", ℓπ_w_unconstrained),
#         ("naive", ℓπ_unconstrained),
#         ("no_synth", ℓπ_ns_unconstrained),
#         ("beta_all", ℓπ_βa_unconstrained),
#         ("noise_aware", ℓπ_na_unconstrained)
#     ])
#
# end


function init_turing_models(
    y_real, X_real, groups_real,
    y_synth, X_synth, groups_synth,
    αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
    nₚ, nₛ, β, βw, w)

    return OrderedDict([
        (
            "beta",
            β_model(
                y_real, X_real, groups_real,
            	y_synth, X_synth, groups_synth,
            	αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
            	nₚ, nₛ, β, βw
            )
        ),
        (
            "weighted",
            weighted_model(
                y_real, X_real, groups_real,
                y_synth, X_synth, groups_synth,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ, w
            )
        ),
        (
            "naive",
            naive_model(
                y_real, X_real, groups_real,
                y_synth, X_synth, groups_synth,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ
            )
        ),
        (
            "no_synth",
            no_synth_model(
                y_real, X_real, groups_real,
                αₚ, βₚ, μₚ, σₚ, νₚ, Σₚ,
                nₚ, nₛ
            )
        ),
    ])

end
