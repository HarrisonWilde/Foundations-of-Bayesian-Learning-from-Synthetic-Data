function evaluate_logistic_samples(X, y, samples, c; plot_roc=false)
    ps = probabilities(X, samples)
    if plot_roc
        plot_roc_curve(y, ps)
    end
    yX = y .* X
    return roc_auc(y, ps, c), log_loss(yX, samples), marginal_likelihood_estimate(yX, samples)
end


function roc_auc(y, ps, c)
    yc = categorical(y)
    levels!(yc, levels(c))
    a = auc([UnivariateFinite(c, [1.0 - p, p]) for p in ps], yc)
    if isnan(a)
        a = 1.
    end
    return a
end


function probabilities(X, samples)
    N = size(samples)[1] + 1
    avg = zeros(size(X)[1])
    for θ in eachrow(samples)
        avg += logistic.(X * θ) ./ N
    end
    return avg
end


function log_loss(yX, samples)
    N = size(samples)[1]
    avg = 0
    for θ in eachrow(samples)
        avg += sum(ℓpdf_BL.(yX * θ)) / N
    end
    return -avg
end


# https://www.jstor.org/stable/pdf/2291091.pdf?refreqid=excelsior%3Ab194b370e4efc9f1d9ae29b7c7c5c6da
function marginal_likelihood_estimate(yX, samples)
    N = size(samples)[1]
    avg = 0
    for θ in eachrow(samples)
        avg += sum(pdf_BL.(yX * θ) .^ -1) / N
    end
    return avg ^ -1
    # mean(map(θ -> sum(pdf_bernoulli_logit.(X_test * θ, y_test)), samples))
end


function evaluate_gaussian_samples(y, dgp, samples, method="Newton")

    post_pdf, post_cdf, inv_post_cdf = setup_posterior_predictive(samples)
    ll = log_loss(y, samples)
    Dkl, _ = kld(post_pdf, dgp, samples)
    Dwass = wass_approx(dgp, samples)
    # @time Dwass2, _ = wassersteind(inv_post_cdf, post_cdf, dgp, samples, method)
    # @show Dwass1, Dwass2
    return Dict(
        "ll" => ll,
        "kld" => Dkl,
        "wass" => Dwass
    )
end


function setup_posterior_predictive(samples)

    function post_pdf(x)
        p = 0
        for (μ, σ²) in eachrow(samples)
            p += pdf_N(μ, √σ², x)
        end
        return p / size(samples)[1]
    end

    function post_cdf(x)
        c = 0
        for (μ, σ²) in eachrow(samples)
            c += cdf_N(μ, √σ², x)
        end
        return c / size(samples)[1]
    end

    function inv_post_cdf(x, method)
        if method == "Newton"
            return find_zero(
                (a -> post_cdf(a) - x,
                a -> post_pdf(a)),
                0.0,
                Roots.Newton()
            )
        elseif method == "Bisection"
            return find_zero(
                a -> post_cdf(a) - x,
                (-1e10, 1e10),
                Roots.Bisection()
            )
        elseif method == "Quantile"
            # THINK THIS IS INCORRECT
            ic = 0
            for (μ, σ²) in eachrow(samples)
                ic += μ + √(2σ²) * erfinv(2x - 1)
            end
            return ic / size(samples)[1]
        elseif method == "FalsePosition"
            return find_zero(
                a -> post_cdf(a) - x,
                (-1e10, 1e10),
                Roots.FalsePosition(1)
            )
        end
    end

    return post_pdf, post_cdf, inv_post_cdf

end
# Approximate posterior predictive by averaging over sample pdfs
# integrate over limits +- 5 * std of DGP


function kld(post_pdf, dgp, samples)
    f(x) = pdf(dgp, x) * log(pdf(dgp, x) / post_pdf(x))
    return quadgk(f, dgp.σ * -5, dgp.σ * 5)
end


function log_loss(y, samples)
    N = size(samples)[1]
    avg = 0
    for (μ, σ²) in eachrow(samples)
        avg += sum(pdf_N.(μ, √σ², y))
    end
    return -log(avg / N)
end


function wass_approx(dgp, samples, n=1000000)
    mixture = MixtureModel(Distributions.Normal[Distributions.Normal(μ, √σ²) for (μ, σ²) in eachrow(samples)])
    inv_dgp_cdf(x) = quantile(dgp, x)
    mix_samples = sort(rand(mixture, n - 1))
    return sum([abs(inv_dgp_cdf(i / n) - mix_samples[i]) for i in 1:(n - 1)]) / (n - 1)
end


function wassersteind(inv_post_cdf, post_cdf, dgp, samples, method)
    inv_dgp_cdf(x) = quantile(dgp, x)
    f(x) = abs(inv_post_cdf(x, method) - inv_dgp_cdf(x))
    Dwass, err = quadgk(f, 0, 1)
    return Dwass, err
end
