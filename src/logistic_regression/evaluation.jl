function evaluate_samples(X, y, samples, c; plot_roc=false)
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
        a = 1
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
