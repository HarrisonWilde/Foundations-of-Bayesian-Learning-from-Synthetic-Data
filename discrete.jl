using Distributions
#using Plots


# think about prior for parameters- maybe also discrete!

r = 5
p = 0.1
n_real = 20
n_realH = 1000
n_eval = 100
dgp = NegativeBinomial(r, p)
real_data = rand(dgp, n_real)
eval_data = rand(dgp, n_eval)
real_dataH = rand(dgp,n_realH)


function geom_mechanism(v,geom_p=0.5)
    res=v+rand(Geometric(geom_p))*(2*(rand()>0.5)-1)
    res * (res > 0) + 0 # hack
end
syn_data = geom_mechanism.(real_dataH)


#  Geometric https://drive.google.com/file/d/114GWEJdlEsHa7wKtKFHEnpG56MVQenlq/view?usp=sharing


prior_params = [(r, p) for r in [1:5;] for p in [0.1, 0.2, 0.3]]
prior_weights = [1.0 for i = 1:length(prior_params)]

logpdf.(dgp, real_data)

function posterior(prior_weights, prior_params, data)
    loglikes = [sum(logpdf.(NegativeBinomial(r, p), data)) for (r, p) in prior_params]
    loglikes = loglikes .- minimum(loglikes)
    posterior_v = prior_weights .* exp.(loglikes)
    posterior_v = posterior_v / sum(posterior_v)
    return posterior_v
end

function do_exp(prior_weights, prior_params, syn_data)
    prior_weights = posterior(prior_weights, prior_params, syn_data)
    return prior_weights
end

function evaluate(posterior, eval_data)
    f(x) = pdf(dgp, x) * log(pdf(dgp, x) / post_pdf(x))
    kld = quadgk(f, dgp.σ * -5, dgp.σ * 5)
    return kld
end

do_exp(prior_weights, prior_params, syn_data)

results = [
    evaluate(posterior(prior_weights, prior_params, syn_data[:i]), eval_data)
    for i in 1:size(syn_data)[1]
]