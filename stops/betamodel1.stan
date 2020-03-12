
data {
   int<lower=0> N;  // number of observations
   int<lower=0> D;  // number of days
   int<lower=0> O;  // number of officers
   int<lower=0> W;  // number of day of week
   int<lower=0> M;  // number of months
   int<lower=0> H;  // number of holidays
   int<lower=0> P;  // number of periods
   int day[N]; 
   int officer[N]; 
   int week[N]; 
   int month[N];
   int holiday[N];
   int period[N];
   int y[N];        // number of tickets written per officer per day
   int z[D];        // number of crashes per day
   int T;
   real beta_w;
   real beta;
}

parameters {
   real beta_0;
   real beta_1[O];
   real beta_2[W];
   real beta_3[M];
   real beta_4[H];
   real beta_5[P];
   vector[D] epsilon;
   
   real<lower=0> sigma;
   real<lower=0> sigma_0;
   real<lower=0> sigma_1;
   real<lower=0> sigma_2;
   real<lower=0> sigma_3;
   real<lower=0> sigma_4;
   real<lower=0> sigma_5;
}

transformed parameters {
    
   real y_int_term; 
   real z_int_term; 
   y_int_term = 0;
   z_int_term = 0;
   for(i in 0:T) {
      y_int_term += exp(poisson_log_lpmf(i | sigma * epsilon[day] + lambda)) ^ (1 + beta);
      z_int_term += exp(poisson_log_lpmf(i | sigma_0 * epsilon + beta_0)) ^ (1 + beta);
   }
   y_int_term = 1.0 / (1.0 + beta) * y_int_term;
   z_int_term = 1.0 / (1.0 + beta) * z_int_term;

}

model {
   vector[N] lambda;
   for(n in 1:N) lambda[n] = sigma_1 * beta_1[officer[n]] + 
                                          sigma_2 * beta_2[week[n]] + 
                                          sigma_3 * beta_3[month[n]] +
                                          sigma_4 * beta_4[holiday[n]] +
                                          sigma_5 * beta_5[period[n]];
                                          
   // Prior
   beta_0  ~ normal(0, 1);
   beta_1  ~ normal(0, 1);
   beta_2  ~ normal(0, 1);
   beta_3  ~ normal(0, 1);
   beta_4  ~ normal(0, 1);
   beta_5  ~ normal(0, 1);
   epsilon ~ normal(0, 1);
   
   sigma   ~ chi_square(1);
   sigma_0 ~ chi_square(1);
   sigma_1 ~ chi_square(1);
   sigma_2 ~ chi_square(1);
   sigma_3 ~ chi_square(1);
   sigma_4 ~ chi_square(1);
   sigma_5 ~ chi_square(1);

   //Likelihood
   target += beta_w * ((1.0 / beta) * exp(poisson_log_lpmf(y | sigma * epsilon[day] + lambda)) ^ (beta) - y_int_term);
   target += beta_w * ((1.0 / beta) * exp(poisson_log_lpmf(z | sigma_0 * epsilon + beta_0)) ^ (beta) - z_int_term);
}
