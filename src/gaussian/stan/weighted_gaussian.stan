// Weighted
data {

   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   vector[n] y1;
   int<lower=0> m;
   vector[m] y2;
   real p_mu;
   real<lower=0> p_alpha;
   real<lower=0> p_beta;
   real<lower=0> hp;
   real<lower=0> scale;
   real beta;
   real beta_w;
   real w;

}

parameters {

   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

model {

   // The prior
   sigma2 ~ inv_gamma(p_alpha, p_beta);
   mu ~ normal(p_mu, sqrt(sigma2) * hp);

   // The likelihood
   target += normal_lpdf(y1 | mu, sqrt(sigma2));
   target += w * normal_lpdf(y2 | mu, sqrt(sigma2));

}
