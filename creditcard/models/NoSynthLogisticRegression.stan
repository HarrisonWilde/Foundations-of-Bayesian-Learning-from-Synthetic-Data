// KLD Logistic Regression (No Synthetic)
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

model {

    // Uninformative priors
    coefs ~ normal(0, 100);
    alpha ~ normal(0, 100);

    // The likelihood
    target += bernoulli_logit_glm_lpmf(y_real | X_real, alpha, coefs);

}
