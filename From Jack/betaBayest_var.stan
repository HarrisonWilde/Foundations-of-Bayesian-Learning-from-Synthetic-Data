
data {
   // Inputs for the sampler: data and prior hyperparameters
   int<lower=0> n;
   matrix[n,1] y;
   real mu_m;
   real<lower=0> mu_s;
   real<lower=0> sig_p1;
   real<lower=0> sig_p2;
   real<lower =0> df;
   real<lower=0> w;
   real beta;
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
  
  int_term = (tgamma((df+1.0)/2.0)^(beta+1.0)*tgamma((beta*df+beta+df)/2.0))/((1.0+beta)*tgamma(df/2)^(beta+1)*tgamma((beta*df+beta+df+1.0)/2.0)*(df)^((beta)/2.0)*pi()^((beta)/2.0)*sigma2^(beta/2.0));
  
  
}

model {
   // The prior
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));


   // The general Bayesian loss function
   for(i in 1:n){
      increment_log_prob(w*((1.0/beta)*exp(student_t_log(y[i,1],df,mu,sqrt(sigma2)))^(beta)-int_term));
   }
}

generated quantities {
   // Sampling from the posterior predictive
   real y_predict;
   y_predict = student_t_rng(df,mu,sqrt(sigma2));

}
