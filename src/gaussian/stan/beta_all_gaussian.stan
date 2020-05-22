// Beta-Div (All)
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
   real lambda;

}

parameters {

   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

transformed parameters {

  // Calculates the integral term 1/(beta+1)int f(z;theta)^(beta+1) dz
  real int_term;
  int_term = (1 / ((2.0 * pi()) ^ (beta / 2.0) * (1 + beta) ^ 1.5 * ((sigma2) ^ (beta / 2))));

}

model {

   // The prior
   sigma2 ~ inv_gamma(p_alpha, p_beta);
   mu ~ normal(p_mu, sqrt(sigma2) * hp);

   // The general Bayesian loss function
   for (i in 1:n) {
      target += beta_w * ((1 / beta) * exp(normal_lpdf(y1[i] | mu, sqrt(sigma2))) ^ (beta) - int_term);
   }

   for (i in 1:m) {
      target += beta_w * ((1 / beta) * exp(normal_lpdf(y2[i] | mu, sqrt(sigma2))) ^ (beta) - int_term);
   }
}
