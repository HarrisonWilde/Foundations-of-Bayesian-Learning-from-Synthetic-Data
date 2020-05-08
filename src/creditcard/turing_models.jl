"""
    WeightedBernoulliLogit(p<:Real, w<:Real)
A univariate bernoulli logit distribution with weight w
"""
struct WeightedBernoulliLogit{T<:Real, U<:Real} <: DiscreteUnivariateDistribution
    logitp::T
	w::U
end

function Distributions.logpdf(d::WeightedBernoulliLogit{<:Real}, k::Int)
    return d.w * logpdf_bernoulli_logit(d.logitp, k)
end

# function Distributions.pdf(d::WeightedBernoulliLogit{<:Real}, k::Int)
#     return pdf_bernoulli_logit(d.logitp, k)
# end


"""
    BetaDBernoulliLogit(p<:Real)
A univariate bernoulli logit distribution beta-diverged
"""
struct BetaDBernoulliLogit{T<:Real, U<:Real, V<:Real} <: DiscreteUnivariateDistribution
    logitp::T
	β::U
	βw::V
end

function Distributions.logpdf(d::BetaDBernoulliLogit{<:Real}, k::Int)
    return d.βw * logpdf_betad_bernoulli_logit(d.logitp, k, d.β)
end



@model logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, w) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ WeightedBernoulliLogit.(X_real * θ, 1)
	y_synth .~ WeightedBernoulliLogit.(X_synth * θ, w)

end

@model β_logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw) = begin

    θ ~ MvNormal(fill(0, θ_dim), σ)
    y_real .~ WeightedBernoulliLogit.(X_real * θ, 1)
	y_synth .~ BetaDBernoulliLogit.(X_synth * θ, β, βw)

end


chn = sample(β_logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, β, βw), Turing.NUTS(), 5000)
chn = sample(logistic_regression(X_real, X_synth, y_real, y_synth, θ_dim, σ, w), Turing.NUTS(), 5000)



"""
SHOULD I BE PRE-ALLOCATING OUTPUTS?
"""


function ∂ℓπ∂θ_kld(σ, w, X_real, y_real, X_synth, y_synth)

    function logpost_and_gradient(θ::Array{Float64,1})
		# 15.788 μs (0.00% GC)  memory estimate:  11.63 KiB  allocs estimate:  1
        z_real = X_real * θ
		# 10.809 μs (0.00% GC)  memory estimate:  3.00 KiB  allocs estimate:  1
        z_synth = X_synth * θ
		# 25.477 ns (5.32% GC)  memory estimate:  16 bytes  allocs estimate:  1
        X_real_T = transpose(X_real)
		# 27.118 ns (5.65% GC)  memory estimate:  16 bytes  allocs estimate:  1
        X_synth_T = transpose(X_synth)

		# 180.215 ns (26.57% GC)  memory estimate:  544 bytes  allocs estimate:  1
        ℓprior = logpdf_centred_mvnormal(σ, θ)
		# 8.541 μs (13.97% GC)  memory estimate:  11.69 KiB  allocs estimate:  4
        ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))
		# 2.792 μs (18.18% GC)  memory estimate:  3.08 KiB  allocs estimate:  5
        ℓsynth = w * sum(logpdf_bernoulli_logit.(z_synth, y_synth))

		# 330.819 ns (28.53% GC)  memory estimate:  1.08 KiB  allocs estimate:  3
        ∂ℓprior∂θ = -θ / abs2(σ)
		# 34.651 μs (8.73% GC)  memory estimate:  25.08 KiB  allocs estimate:  17
        ∂ℓreal∂θ = *(X_real_T, @. y_real * logistic(-z_real)) - *(X_real_T, @. (1.0 - y_real) * logistic(z_real))
		# CRASHES ?
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
		# 16.351 μs (0.00% GC)  memory estimate:  11.63 KiB  allocs estimate:  1
		z_real = X_real * θ
		# 10.505 μs (8.47% GC)  memory estimate:  3.00 KiB   allocs estimate:  1
    	z_synth = X_synth * θ
		# 29.516 ns (6.31% GC)  memory estimate:  16 bytes   allocs estimate:  1
    	X_real_T = transpose(X_real)

		# 203.713 ns (33.19% GC)  memory estimate:  544 bytes  allocs estimate:  1
    	ℓprior = logpdf_centred_mvnormal(σ, θ)
		# 9.153 μs (14.08% GC)    memory estimate:  11.69 KiB  allocs estimate:  4
		ℓreal = sum(logpdf_bernoulli_logit.(z_real, y_real))

		# 2.310 μs (0.00% GC)  memory estimate:  3.05 KiB  allocs estimate:  3
        pdf_synth = pdf_bernoulli_logit.(z_synth, y_synth)
		# 1.631 μs (37.36% GC)  memory estimate:  3.03 KiB  allocs estimate:  3
		logistic_z = logistic.(z_synth)
		# 17.577 μs (0.00% GC)  memory estimate:  3.73 KiB  allocs estimate:  29
		ℓsynth = βw * sum(@. (1.0 / β) * (
                pdf_synth
            ) ^ β - (1.0 / (β + 1.0)) * (
                logistic_z ^ (β + 1.0)
                + (1.0 - logistic_z) ^ (β + 1.0)
            )
        )

		# 383.653 ns (35.74% GC)  memory estimate:  1.08 KiB  allocs estimate:  3
        ∂ℓprior∂θ = -θ / abs2(σ)
		# 35.578 μs (6.71% GC)   memory estimate:  25.08 KiB  allocs estimate:  17
        ∂ℓreal∂θ = *(X_real_T, @. y_real * logistic(-z_real)) - *(X_real_T, @. (1.0 - y_real) * logistic(z_real))
		# 77.649 μs (5.46% GC)   memory estimate:  166.91 KiB  allocs estimate:  6
        ∂logistic_zX = @. ∂logistic(z_synth) * X_synth
		# 92.983 μs (5.72% GC)   memory estimate:  167.08 KiB  allocs estimate:  14
        ∂ℓpdf_synth∂θ = @. y_synth * logistic(-z_synth) * X_synth - (1.0 - y_synth) * logistic_z * X_synth
		# 420.226 μs (1.31% GC)  memory estimate:  168.41 KiB  allocs estimate:  25
		∂ℓsynth∂θ = vec(βw * sum((@. pdf_synth ^ β * (
                ∂ℓpdf_synth∂θ
            ) - (
                logistic_z ^ β * ∂logistic_zX - (1.0 - logistic_z) ^ β * ∂logistic_zX
            )),
            dims=1
        ))
        return (ℓprior + ℓreal + ℓsynth), (∂ℓprior∂θ + ∂ℓreal∂θ + ∂ℓsynth∂θ)
    end

    return logpost_and_gradient
end
