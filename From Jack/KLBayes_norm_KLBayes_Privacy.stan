
data {
   
   int<lower=0> n;
   matrix[n,1] data_obs_H1;
   int<lower=0> m;
   matrix[m,1] data_obs_H2;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;

}

   
parameters 
{
   
   real mu;
   real<lower=0> sigma2;

}



model {
  
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));
  
   data_obs_H1[,1] ~ normal(mu, sqrt(sigma2));
   data_obs_H2[,1] ~ normal(mu, sqrt(sigma2));
  
}


