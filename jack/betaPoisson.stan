
data {
   
   int<lower=0> n;
   int y[n];
   real<lower=0> a_0;
   real<lower=0> b_0;
   real<lower=0> w;
   real<lower=0> beta;
   int<lower=0> T;

}

parameters {

   real<lower=0> lambda;

}

transformed parameters {
   
   real int_term; 
   int_term = 0;
   for(i in 1:(T+1)){
      int_term  += exp(poisson_lpmf(i-1|lambda))^(1+beta);
   }
   int_term = 1.0/(1.0+beta)*int_term;
}

model {

   lambda ~ gamma(a_0,b_0);
   for(i in 1:n){
      target += w*(1.0/beta*(exp(poisson_lpmf(y[i]|lambda)))^beta-int_term);
   }
}

generated quantities {
   real y_predict[n];
   for(i in 1:n){
      y_predict[i] = poisson_rng(lambda);
   }
}
