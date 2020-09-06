// Noise Aware (NL Convolution)
functions {
   // Need to write vectorised normal lpdf functions.
   real normal_laplace_pdf(real y, real mu, real sigma, real lambda) {
      return 1 / (2 * lambda) * (
         exp((mu - y) / lambda + sigma ^ 2 / (2 * lambda ^ 2)) *
         std_normal_cdf(((y - mu) / sigma - sigma / lambda)) +
         exp((y - mu) / lambda + sigma ^ 2 / (2 * lambda ^ 2)) *
         (1 - std_normal_cdf(((y - mu) / sigma + sigma / lambda)))
      );
   }

}

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
   real<lower=0> lambda;

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
   for (i in 1:m) {
      target += log(normal_laplace_pdf(y2[i], mu, sqrt(sigma2), lambda));
   }

}
