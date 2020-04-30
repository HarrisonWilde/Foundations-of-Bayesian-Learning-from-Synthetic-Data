
function ℓπ_kld(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost(θ::Array{Float64,1})
        D = size(θ)[1]
        ℓprior = sum(logpdf(MvNormal(D, σ), θ))
        ℓreal = sum(logpdf.(BinomialLogit.(1, X_real * θ), y_real))
        ℓsynth = w * sum(logpdf.(BinomialLogit.(1, X_synth * θ), y_synth))
        return (ℓprior + ℓreal + ℓsynth)
    end

    return logpost
end


function ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
        D = size(θ)[1]
        Σ = abs2(σ) * I + zeros(D, D)
        z_real = X_real * θ
        z_synth = X_synth * θ
        ℓprior = sum(logpdf(MvNormal(D, σ), θ))
        ℓreal = sum(logpdf.(BinomialLogit.(1, z_real), y_real))
        ℓsynth = w * sum(logpdf.(BinomialLogit.(1, X_synth * θ), y_synth))
        ∂ℓprior∂θ = -inv(Σ) * θ
        ∂ℓreal∂θ = vec(transpose(y_real .* logistic.(-z_real)) * X_real - transpose((1.0 .- y_real) .* logistic.(z_real)) * X_real)
        ∂ℓsynth∂θ = vec(w .* transpose(y_synth .* logistic.(-z_synth)) * X_synth - transpose((1.0 .- y_synth) .* logistic.(z_synth)) * X_synth)
        return (ℓprior + ℓreal + ℓsynth), (∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end


function ℓπ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)

    function logpost(θ::Array{Float64,1})
        D = size(θ)[1]
        z_real = X_real * θ
        z_synth = X_synth * θ
        ℓprior = sum(logpdf(MvNormal(D, σ), θ))
        ℓreal = sum(logpdf.(BinomialLogit.(1, z_real), y_real))
        ℓsynth = βw * sum(
            (1 / β) * (
                y_synth .* logistic.(X_synth * θ) + (1 .- y_synth) .* (1 .- logistic.(X_synth * θ))
            ) .^ β - (1 / (β + 1)) .* (
                logistic.(X_synth * θ) .^ (β + 1)
                .+ (1 .- logistic.(X_synth * θ)) .^ (β + 1)
            )
        )
        return (ℓprior + ℓreal + ℓsynth)
    end

    return logpost
end


function ∂ℓπ∂θ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
        D = size(θ)[1]
        Σ = abs2(σ) * I + zeros(D, D)
        z_real = X_real * θ
        z_synth = X_synth * θ
        ℓprior = sum(logpdf(MvNormal(D, σ), θ))
        ℓreal = sum(logpdf.(BinomialLogit.(1, z_real), y_real))
        ℓsynth = βw * sum(
            (1 / β) * (
                y_real .* StatsFuns.logistic.(X_synth * θ) + (1 .- y_real) .* (1 .- StatsFuns.logistic.(X_synth * θ))
            ) .^ β - (1 / (β + 1)) .* (
                StatsFuns.logistic.(X_synth * θ) .^ (β + 1)
                .+ (1 .- StatsFuns.logistic.(X_synth * θ)) .^ (β + 1)
            )
        )
        ∂ℓprior∂θ = -inv(Σ) * θ
        ∂ℓreal∂θ = vec(transpose(y_real .* logistic.(-z_real)) * X_real - transpose((1.0 .- y_real) .* logistic.(z_real)) * X_real)
        ∂ℓsynth∂θ = vec(βw * sum(
            (
                (y_synth .* logistic.(z_synth) + (1 .- y_synth) .* (1 .- logistic.(z_synth))) .^ (β - 1)
                .* (∂logistic.(z_synth) .* X_synth .* y_synth - ∂logistic.(z_synth) .* X_synth .* (1 .- y_synth))
                .- (logistic.(z_synth) .^ β .* ∂logistic.(z_synth) .* X_synth .- (1 .- logistic.(z_synth)) .^ β .* ∂logistic.(z_synth) .* X_synth)
            ),
            dims=1
        ))
        return (ℓprior + ℓreal + ℓsynth), (∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end


"""
FUNCTIONS BELOW SHOULD BE IDENTICAL BUT UTILISE THE @. MACRO FOR A SLIGHT PERFORMANCE BOOST
"""


function ℓπ_kld_opt(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost(θ::Array{Float64,1})
        z_real = X_real * θ
        z_synth = X_synth * θ

        ℓprior = logpdf_centred_mvnormal(σ, θ)
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))
        ℓsynth = w * sum(logpdf_bernoulli_logit.(z_synth, y_synth))

        return (ℓprior + ℓreal + ℓsynth)
    end

    return logpost
end


function ∂ℓπ∂θ_kld_opt(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
        z_real = X_real * θ
        z_synth = X_synth * θ
        X_real_T = transpose(X_real)
        X_synth_T = transpose(X_synth)

        ℓprior = logpdf_centred_mvnormal(σ, θ)
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))
        ℓsynth = w * sum(logpdf_bernoulli_logit.(z_synth, y_synth))

        ∂ℓprior∂θ = -θ / abs2(σ)
        ∂ℓreal∂θ = @. $*(X_real_T, y_real * logistic(-z_real)) - $*(X_real_T, (1.0 - y_real) * logistic(z_real))
        ∂ℓsynth∂θ = w * (@. $*(X_synth_T, y_synth * logistic(-z_synth)) - $*(X_synth_T, (1.0 - y_synth) * logistic(z_synth)))

        return (ℓprior + ℓreal + ℓsynth), vec(∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end


function ℓπ_beta_opt(σ, β, βw, X_real, y_real, X_synth, y_synth)

    function logpost(θ::Array{Float64,1})
        z_real = X_real * θ
        z_synth = X_synth * θ

        ℓprior = logpdf_centred_mvnormal(σ, θ)
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))

        logistic_z = logistic.(z_synth)
        ℓsynth = βw * sum(@. (1.0 / β) * (
                pdf_bernoulli_logit(z_synth, y_synth)
            ) ^ β - (1.0 / (β + 1.0)) * (
                logistic_z ^ (β + 1.0)
                + (1.0 - logistic_z) ^ (β + 1.0)
            )
        )

        return (ℓprior + ℓreal + ℓsynth)
    end

    return logpost
end


function ∂ℓπ∂θ_beta_opt(σ, β, βw, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
        z_real = X_real * θ
        z_synth = X_synth * θ
        X_real_T = transpose(X_real)

        ℓprior = logpdf_centred_mvnormal(σ, θ)
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))

        pdf_synth = pdf_bernoulli_logit.(z_synth, y_synth)
        logistic_z = logistic.(z_synth)
        ℓsynth = βw * sum(@. (1.0 / β) * (
                pdf_synth
            ) ^ β - (1.0 / (β + 1.0)) * (
                logistic_z ^ (β + 1.0)
                + (1.0 - logistic_z) ^ (β + 1.0)
            )
        )

        ∂ℓprior∂θ = -θ / abs2(σ)
        ∂ℓreal∂θ = @. $*(X_real_T, y_real * logistic(-z_real)) - $*(X_real_T, (1.0 - y_real) * logistic(z_real))

        ∂logistic_zX = @. ∂logistic(z_synth) * X_synth
        # ∂ℓsynth∂θ_old = βw * sum((@. pdf_synth ^ (β - 1.0) * (
        #         ∂logistic_zX * y_synth - ∂logistic_zX * (1.0 - y_synth)
        #     ) - (
        #         logistic_z ^ β * ∂logistic_zX - (1.0 - logistic_z) ^ β * ∂logistic_zX
        #     )),
        #     dims=1
        # )
        ∂ℓpdf_synth∂θ = @. y_synth * logistic(-z_synth) * X_synth - (1.0 - y_synth) * logistic_z * X_synth
        ∂ℓsynth∂θ = βw * sum((@. pdf_synth ^ β * (
                ∂ℓpdf_synth∂θ
            ) - (
                logistic_z ^ β * ∂logistic_zX - (1.0 - logistic_z) ^ β * ∂logistic_zX
            )),
            dims=1
        )
        return (ℓprior + ℓreal + ℓsynth), vec(∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end
