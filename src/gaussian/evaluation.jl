function evaluate_samples(y, dgp, samples, method="Newton")

    post_pdf, post_cdf, inv_post_cdf = setup_posterior_predictive(samples)
    # mixture = MixtureModel(Distributions.Normal[Distributions.Normal(μ, √σ²) for (μ, σ²) in eachrow(samples)])
    ll = log_loss(y, samples)
    @time Dkl, _ = kld(post_pdf, dgp, samples)
    @time Dwass, _ = wassersteind(inv_post_cdf, post_cdf, dgp, samples, method)
    return ll, Dkl, Dwass
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
        avg += sum(ℓpdf_N.(μ, √σ², y)) / N
    end
    return -avg
end


# function quantile_bisect(post_cdf, p::Real, lx::Real=-1e8, rx::Real=1e8, tol::Real=1.0e-12)
#
#     # find quantile using bisect algorithm
#     cl = post_cdf(lx)
#     cr = post_cdf(rx)
#     @assert cl <= p <= cr
#     while rx - lx > tol
#         m = (lx + rx)/2
#         c = post_cdf(m)
#         if p > c
#             cl = c
#             lx = m
#         else
#             cr = c
#             rx = m
#         end
#     end
#     return (lx + rx)/2
# end


function wassersteind(inv_post_cdf, post_cdf, dgp, samples, method)
    inv_dgp_cdf(x) = quantile(dgp, x)
    f(x) = abs(inv_post_cdf(x, method) - inv_dgp_cdf(x))
    @time Dwass, err = quadgk(f, 0, 1)
    return Dwass, err
end


# g(x) = abs(find_zero((a -> post_cdf(a) - x, a -> post_pdf(a)), 0.0, Roots.Newton()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero((a -> post_cdf(a) - x, a -> post_pdf(a), a -> ForwardDiff.derivative(post_pdf, float(a))), 0.0, Roots.Halley()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order16()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order8()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order5()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order2()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order1()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
# g(x) = abs(find_zero(a -> post_cdf(a) - x, 0.0, Roots.Order0()) - inv_dgp_cdf(x))
# @time quadgk(g, 0, 1)
