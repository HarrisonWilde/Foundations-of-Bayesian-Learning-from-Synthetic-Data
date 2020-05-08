

int_term = (1 / ((2.0 * π) ^ (β / 2.0) * (1.0 + β) ^ 1.5 * ((σ²) ^ (β / 2))))

model {

   σ² ~ inv_gamma(α₁, α₂);
   μ ~ normal(μₚ, sqrt(σ²) * hp);

   target += normal_lpdf(y1 | mu, sqrt(sigma2));
   // The general Bayesian loss function
   for (i in 1:m) {
      target += beta_w * ((1 / beta) * exp(normal_lpdf(y2[i] | mu, sqrt(sigma2))) ^ (beta) - int_term);
   }
}


using Pkg

function normal_pdf(μ, σ, y)
    1 / (σ * √(2π)) * exp(-1/2 * ((y - μ) / σ) ^ 2)
end

function gauss_pdf_beta(μ, σ, y, β, βw)
    int_term = (1.0 / ((2.0 * π) ^ (β / 2.0) * (1.0 + β) ^ 1.5 * (σ ^ (β / 2.0))))
    βw * ((1.0 / β) * normal_pdf(μ, σ, y) ^ (β) - int_term)
end

@model gaussian_beta(y, α, θ, μₚ, hp)

    σ² ~ InverseGamma(α, θ)
    μ ~ Normal(μₚ, σ² * hp)
    DynamicPPL.getlogp(_varinfo) += gauss_pdf_beta(μ, σ, y, β, βw)

end


function gaussbeta_posterior(μ, θ, σ²; prior args)



end

# gauss_pdf_beta(μ, σ, y, β, βw)#
#
#
# # data generating mechanism
# mu_star=  1.0
# sigma_star= 2.0
#
# muWidth=10.0
# sigmaWidth=10.0
#
# #normalising constant
# nc=hcubature(x->x upd[x,mu,sigma], [-GaussWidth,0],[GaussWidth,sigmaWidth],)
#
#
# #logloss on test set
#
# function logpdf_gaussian(x,mu,sigma)
#     logpdf(NormalDistribution(mu,sigma),x)
# end
# ### log loss on withhold
# logloss= hcubature( logpdf_gaussian(x,mu,sigma)  upd(mu,sigma)/nc, [-GaussWidth,0],[GaussWidth,sigmaWidth],)

# log loss expectation
logp (x given theta)
logloss=hcubature(upd[mu,sigma], [-GaussWidth,0],[GaussWidth,sigmaWidth],)

Pkg.add("QuadG")

using QuadGK
integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)


normc=quadgk (x-> upos_dens(x,y,beta),-20,20,rol´

logloss= quadgk pos_dens/normc pdf(Normal)
