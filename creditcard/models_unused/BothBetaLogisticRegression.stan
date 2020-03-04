# BetaD-Both Logistic Regression
data {
    
    // Inputs for the sampler: data and prior hyperparameters
    int<lower=0> f;
    int<lower=0> a;
    matrix[a, f] X_real;
    int<lower=0, upper=1> y_real[a];
    int<lower=0> b;
    matrix[b, f] X_synth;
    int<lower=0, upper=1> y_synth[b];
    int<lower=0> c;
    matrix[c, f] X_test;
    int<lower=0, upper=1> y_test[c];
    real<lower=0> w;
    real beta;
    real<lower=0> beta_w;

}

parameters {
    
    // Parameters for which we do inference
    vector[f] coefs;
    real alpha;

}

transformed parameters {
    
    // Calculates the integral term 1/(beta+1)int f(z;theta)^(beta+1) dz
    real int_term;
    int_term = (1 / ((2.0 * pi()) ^ (beta / 2.0) * (1 + beta) ^ 1.5 * ((sigma2) ^ (beta / 2))));

}

model {
    
    // The likelihood
    target += beta_w * ((1 / beta) * exp(bernoulli_logit_lpmf(y_real | alpha + coefs * X_real)) ^ (beta) - int_term);
    target += beta_w * ((1 / beta) * exp(bernoulli_logit_lpmf(y_synth | alpha + coefs * X_synth)) ^ (beta) - int_term);

}

generated quantities {
    
    // Calculating log likelihoods given iters theta for each y tilde and unseen
    vector[c] predictions;
    predictions = bernoulli_logit_rng(alpha + coefs * X_test);

}
