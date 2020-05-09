// Naive
data {

   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   vector[n] y1;
   int<lower=0> m;
   vector[m] y2;
   int<lower=0> j;
   vector[j] y_unseen;
   int<lower=0> k;
   vector[k] y_tildes;
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
   target += normal_lpdf(y2 | mu, sqrt(sigma2));

}

generated quantities {

   // Sampling from the posterior predictive
   vector[k] log_likes_tildes;
   vector[j] log_likes_unseen;
   for (i in 1:k) {
      log_likes_tildes[i] = normal_lpdf(y_tildes[i] | mu, sqrt(sigma2));
   }
   for (i in 1:j) {
      log_likes_unseen[i] = normal_lpdf(y_unseen[i] | mu, sqrt(sigma2));
   }

}
