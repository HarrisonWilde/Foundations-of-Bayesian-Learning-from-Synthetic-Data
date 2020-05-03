
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
}

parameters 
{
   // Parameters for which we do inference
   real mu;
   real<lower=0> sigma2;

}

model {
   // The prior
   sigma2 ~ inv_gamma(sig_p1,sig_p2);
   mu ~ normal(mu_m,sqrt(sigma2*mu_s));
   
   // The likelihood 
   y[,1] ~ student_t(df,mu,sqrt(sigma2));

}

generated quantities {
   // Sampling from the posterior predictive
   real y_predict;
   y_predict = student_t_rng(df,mu,sqrt(sigma2));

}
