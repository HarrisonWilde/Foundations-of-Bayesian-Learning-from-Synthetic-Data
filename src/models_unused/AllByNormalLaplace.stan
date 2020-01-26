
functions {
   
   real log_mills_ratio(real z) {
      return log1m(Phi(z)) - std_normal_lpdf(z); 
   }

   real normal_laplace_lpdf(vector y, real mu, real sigma, real scale) {
      real total = 0.0;
      real k = (sqrt(2) * sigma) / scale;
      for (i in 1:num_elements(y)) {
         real r = (y[i] - mu) / sigma;
         total += -log(sqrt(2) * scale) + std_normal_lpdf(r) + log_sum_exp(log_mills_ratio(k - r), log_mills_ratio(k + r));
      }
      return total;
   }

}

data {
   
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   vector[n] y;
   int<lower=0> j;
   vector[j] y_unseen;
   int<lower=0> k;
   vector[k] y_tildes;
   real mu_m;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> scale;
   real<lower=0> hp;

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
   target += normal_laplace(y | mu, sqrt(sigma2), scale);

}

generated quantities {
   
   // Calculating log likelihoods given iters theta for each y tilde and unseen
   vector[k] log_likes_tildes;
   vector[j] log_likes_unseen;

   for (i in 1:k) {
      log_likes_tildes[i] = normal_lpdf(y_tildes[i] | mu, sqrt(sigma2));
   }
   for (i in 1:j) {
      log_likes_unseen[i] = normal_lpdf(y_unseen[i] | mu, sqrt(sigma2));
   }

}
