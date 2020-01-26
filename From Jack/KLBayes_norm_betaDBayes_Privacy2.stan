
data {
   
   int<lower=0> n;
   matrix[n,1] data_obs_H1;
   int<lower=0> m;
   matrix[m,1] data_obs_H2;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower=0> w;
   real beta;

}

   
parameters  {
   
   real mu;
   real<lower=0> sigma2;

}

transformed parameters {
  
  real int_term;
  
  int_term = (1/((2.0*pi())^(beta/2.0)*(1+beta)^1.5*(sigma2^(beta/2))));
  
  
}


model {
  
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));
  
   for(i in 1:n){
      target += w*((1/beta)*exp(normal_lpdf(data_obs_H1[i,1]| mu,sqrt(sigma2)))^(beta)-int_term);
   }
   data_obs_H2[,1] ~ normal(mu, sqrt(sigma2));
  
}


