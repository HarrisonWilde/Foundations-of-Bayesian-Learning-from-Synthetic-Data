function evalu(X_test, y_test, samples; plot_roc=false)
    ps = probabilities(X_test, samples)
    if plot_roc
        plot_roc_curve(y_test, ps)
    end
    return roc_auc(y_test, ps), log_loss(X_test, y_test, samples), marginal_likelihood_estimate(X_test, y_test, samples)
end


function roc_auc(ys, ps)
    auc([UnivariateFinite(categorical([0, 1]), [1.0 - p, p]) for p in ps], categorical(ys))
end


function probabilities(X, samples)
    N = size(samples)[1]
    avg = logistic.(X * samples[1]) ./ N
    for θ in samples[2:end]
        avg += logistic.(X * θ) ./ N
    end
    return avg
end


function log_loss(X, y, samples)
    N = size(samples)[1]
    avg = 0
    for θ in samples
        avg += sum(logpdf_bernoulli_logit.(X * θ, y)) / N
    end
    return -avg
end


# https://www.jstor.org/stable/pdf/2291091.pdf?refreqid=excelsior%3Ab194b370e4efc9f1d9ae29b7c7c5c6da
function marginal_likelihood_estimate(X, y, samples)
    N = size(samples)[1]
    avg = 0
    for θ in samples
        avg += sum(pdf_bernoulli_logit.(X * θ, y) .^ -1) / N
    end
    return avg ^ -1
    # mean(map(θ -> sum(pdf_bernoulli_logit.(X_test * θ, y_test)), samples))
end
