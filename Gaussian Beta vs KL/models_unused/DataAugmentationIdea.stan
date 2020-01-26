
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
   real<lower=0> scale;
   real<lower=0> hp;

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
  
   sigma2 ~ inv_gamma(sig_p1, sig_p2);
   mu ~ normal(mu_m, sqrt(sigma2 * hp));
   eps_private ~ double_exponential(0, scale);
  
   target += normal_lpdf(y1 | mu, sqrt(sigma2));
   target += normal_lpdf(contam_y2 | mu, sqrt(sigma2));
  
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

