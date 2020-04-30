
function ℓπ_kld(σ, w, X_real, y_real, X_synth, y_synth)

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


function ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
        z_real = X_real * θ
        z_synth = X_synth * θ
        X_real_T = transpose(X_real)
        X_synth_T = transpose(X_synth)

        ℓprior = logpdf_centred_mvnormal(σ, θ)
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))
        ℓsynth = w * sum(logpdf_bernoulli_logit.(z_synth, y_synth))

        ∂ℓprior∂θ = -θ / abs2(σ)
        ∂ℓreal∂θ = *(X_real_T, @. y_real * logistic(-z_real)) - *(X_real_T, @. (1.0 - y_real) * logistic(z_real))
        ∂ℓsynth∂θ = w * (@. $*(X_synth_T, y_synth * logistic(-z_synth)) - $*(X_synth_T, (1.0 - y_synth) * logistic(z_synth)))

        return (ℓprior + ℓreal + ℓsynth), vec(∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end


function ℓπ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)

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


function ∂ℓπ∂θ_beta(σ, β, βw, X_real, y_real, X_synth, y_synth)

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
        ∂ℓreal∂θ = *(X_real_T, @. y_real * logistic(-z_real)) - *(X_real_T, @. (1.0 - y_real) * logistic(z_real))

        ∂logistic_zX = @. ∂logistic(z_synth) * X_synth
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
