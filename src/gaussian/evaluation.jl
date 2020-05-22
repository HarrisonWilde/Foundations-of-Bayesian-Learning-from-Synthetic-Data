function evaluate_samples(y, dgp, samples)

    ll = log_loss(y, samples)
    Dkl = kld(dgp, samples)
    # Dwass = wasserstein_d(dgp, samples)
    return ll, Dkl
end

function setup_posterior_predictive(samples)

    function pp(x)
        p = 0
        for (μ, σ²) in eachrow(samples)
            p += pdf_N(μ, √σ², x)
        end
        return p / size(samples)[1]
    end
    return pp

end
# Approximate posterior predictive by averaging over sample pdfs
# integrate over limits +- 5 * std of DGP
function kld(dgp, samples)
    pp = setup_posterior_predictive(samples)
    f(x) = pdf(dgp, x) * log(pdf(dgp, x) / pp(x))
    Dkl, err = quadgk(f, dgp.σ * -5, dgp.σ * 5)
    return Dkl
end


function log_loss(y, samples)
    N = size(samples)[1]
    avg = 0
    for (μ, σ²) in eachrow(samples)
        avg += sum(ℓpdf_N.(μ, √σ², y)) / N
    end
    return -avg
end


# function wasserstein_distance(dgp, samples)
#     N = size(samples)[1]
#     avg = 0
#     for (μ, σ²) in eachrow(samples)
#         avg += WassersteinD(dgp, Distributions.Normal(μ, √σ²)) / N
#     end
#     return avg
# end
