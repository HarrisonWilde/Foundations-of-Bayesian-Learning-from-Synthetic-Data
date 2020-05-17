// Noise-Aware (N-Laplace Distrib.)
functions {

   real log_mills_ratio(real z) {

      return log1m(Phi(z)) - std_normal_lpdf(z);

   }

   real normal_laplace_lpdf(vector y, real mu, real sigma, real scale) {

      real total = 0.0;
      real k = (sigma) / scale;
      for (i in 1:num_elements(y)) {
         real r = (y[i] - mu) / sigma;
         real thing = -log(2 * scale) + std_normal_lpdf(r) + log_sum_exp(log_mills_ratio(k - r), log_mills_ratio(k + r));
         if (thing < -1e30) {
            print("k: ", k, ", r: ", r, ", thing: ", thing);
         }
         total += thing;
      }
      return total;

   }

}

data {

   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   vector[n] y1;
   int<lower=0> m;
   vector[m] y2;
   real mu_m;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
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
   sigma2 ~ inv_gamma(sig_p1, sig_p2);
   mu ~ normal(mu_m, sqrt(sigma2) * hp);

   // The likelihood
   target += normal_lpdf(y1 | mu, sqrt(sigma2));
   target += normal_laplace_lpdf(y2 | mu, sqrt(sigma2), scale);

}
