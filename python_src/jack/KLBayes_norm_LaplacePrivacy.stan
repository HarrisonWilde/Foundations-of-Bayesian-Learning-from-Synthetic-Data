
data {
   
   int<lower=0> n;
   matrix[n,1] data_obs;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> sensitivity;

}

   
parameters 
{
   
   real mu;
   real<lower=0> sigma2;
   vector[n] eps_private;

}




transformed parameters
{
   
   vector[n] data_true;
   data_true = data_obs[,1] - eps_private; 
}

model {
  
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));
   eps_private ~ double_exponential(0,sensitivity);
  
   data_true ~ normal(mu,sqrt(sigma2));
  
}


