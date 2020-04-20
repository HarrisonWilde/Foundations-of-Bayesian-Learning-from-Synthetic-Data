
data {
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   matrix[n,1] y;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> w;
   real beta;
   real<lower=0> sigma2_adj;
}

parameters 
{
   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

transformed parameters
{
  // Calculates the integral term 1/(beta+1)int f(z;theta)^(beta+1) dz
  real int_term;
  
  int_term = (1/((2.0*pi())^(beta/2.0)*(1+beta)^1.5*((sigma2*sigma2_adj)^(beta/2))));
  
  
}

model {
   // The prior
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));

   // The general Bayesian loss function
   for(i in 1:n){
    increment_log_prob(w*((1/beta)*exp(normal_log(y[i,1],mu,sqrt(sigma2_adj*sigma2)))^(beta)-int_term));
   }
}

generated quantities {
   // Sampling from the posterior predictive
   real y_predict;
   y_predict = normal_rng(mu,sqrt(sigma2*sigma2_adj));

}
