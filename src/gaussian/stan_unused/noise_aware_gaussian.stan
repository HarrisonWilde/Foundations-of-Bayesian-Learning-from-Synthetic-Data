// Noise-Aware (Data Aug.)
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

   real mu;
   real<lower=0> sigma2;
   vector[m] eps_private;

}

transformed parameters {

   vector[m] contam_y2;
   contam_y2 = y2 - eps_private;

}

model {

   sigma2 ~ inv_gamma(p_alpha, p_beta);
   mu ~ normal(p_mu, sqrt(sigma2) * hp);
   eps_private ~ double_exponential(0, scale);

   target += normal_lpdf(y1 | mu, sqrt(sigma2));
   target += normal_lpdf(contam_y2 | mu, sqrt(sigma2));

}
